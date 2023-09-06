from dataclasses import dataclass
from typing import Dict, Type

import torch
from black import Tuple
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import (
    CompositeSpec,
    ReplayBuffer,
    TensorDictReplayBuffer,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.replay_buffers import PrioritizedSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import AdditiveGaussianWrapper, ProbabilisticActor, TanhDelta
from torchrl.objectives import (
    ClipPPOLoss,
    DDPGLoss,
    LossModule,
    SoftUpdate,
    ValueEstimators,
)
from torchrl.objectives.utils import TargetNetUpdater

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig
from benchmarl.utils import DEVICE_TYPING


class Maddpg(Algorithm):
    def __init__(
        self,
        share_param_actor: bool = True,
        share_param_critic: bool = True,
        loss_function: str = "l2",
        delay_value: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.share_param_actor = share_param_actor
        self.share_param_critic = share_param_critic
        self.delay_value = delay_value
        self.loss_function = loss_function

    #############################
    # Overridden abstract methods
    #############################

    def _get_replay_buffer(
        self,
        group: str,
        memory_size: int,
        sampling_size: int,
        traj_len: int,
        storing_device: DEVICE_TYPING,
    ) -> ReplayBuffer:
        return TensorDictReplayBuffer(
            storage=LazyTensorStorage(memory_size, device=storing_device),
            sampler=PrioritizedSampler(
                max_capacity=memory_size,
                alpha=self.experiment_config.off_policy_prioritised_alpha,
                beta=self.experiment_config.off_policy_prioritised_beta,
            ),
            batch_size=sampling_size,
            priority_key=(group, "td_error"),
        )

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, TargetNetUpdater]:
        if continuous:
            # Loss
            loss_module = DDPGLoss(
                actor_network=policy_for_loss,
                value_network=self.get_value_module(group),
                delay_value=self.delay_value,
                loss_function=self.loss_function,
            )
            loss_module.set_keys(
                state_action_value=(group, "state_action_value"),
                reward=(group, "reward"),
                priority=(group, "td_error"),
                done=(group, "done"),
            )

            loss_module.make_value_estimator(
                ValueEstimators.TD0, gamma=self.experiment_config.gamma
            )
            target_net_updater = SoftUpdate(
                loss_module, tau=self.experiment_config.polyak_tau
            )
            return loss_module, target_net_updater
        else:
            raise NotImplementedError(
                "MADDPG is not compatible with discrete actions yet"
            )

    def _get_optimizers(
        self, group: str, loss: ClipPPOLoss, lr: float
    ) -> Dict[str, torch.optim.Optimizer]:

        return {
            "loss_actor": torch.optim.Adam(
                list(loss.actor_network_params.flatten_keys().values()), lr=lr
            ),
            "loss_value": torch.optim.Adam(
                list(loss.value_network_params.flatten_keys().values()), lr=lr
            ),
        }

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        if continuous:
            n_agents = len(self.group_map[group])
            logits_shape = list(self.action_spec[group, "action"].shape)
            actor_input_spec = CompositeSpec(
                {
                    group: CompositeSpec(
                        {
                            "observation": self.observation_spec[group]["observation"]
                            .clone()
                            .to(self.device)
                        },
                        shape=(n_agents,),
                    )
                }
            )
            actor_output_spec = CompositeSpec(
                {
                    group: CompositeSpec(
                        {"param": UnboundedContinuousTensorSpec(shape=logits_shape)},
                        shape=(n_agents,),
                    )
                }
            )
            actor_module = model_config.get_model(
                input_spec=actor_input_spec,
                output_spec=actor_output_spec,
                agent_group=group,
                input_has_agent_dim=True,
                n_agents=n_agents,
                centralised=False,
                share_params=self.share_param_actor,
                device=self.device,
            )

            policy = ProbabilisticActor(
                module=actor_module,
                spec=self.action_spec,
                in_keys=[(group, "param")],
                out_keys=[(group, "action")],
                distribution_class=TanhDelta,
                distribution_kwargs={
                    "min": self.action_spec[(group, "action")].space.minimum,
                    "max": self.action_spec[(group, "action")].space.maximum,
                },
                return_log_prob=False,
            )
            return policy
        else:
            raise NotImplementedError(
                "MADDPG is not compatible with discrete actions yet"
            )

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        return AdditiveGaussianWrapper(
            policy_for_loss,
            annealing_num_steps=self.experiment_config.exploration_annealing_num_frames,
            action_key=(group, "action"),
            # sigma_init = 1.0,
            # sigma_end = 0.1,
        )

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )

        if nested_reward_key not in keys:
            batch.set(
                nested_reward_key,
                batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
            )

        return batch

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False

    @staticmethod
    def on_policy() -> bool:
        return False

    #####################
    # Custom new methods
    #####################

    def get_value_module(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        modules = []

        if self.share_param_critic:
            critic_output_spec = CompositeSpec(
                {"state_action_value": UnboundedContinuousTensorSpec(shape=(1,))}
            )
        else:
            critic_output_spec = CompositeSpec(
                {
                    group: CompositeSpec(
                        {
                            "state_action_value": UnboundedContinuousTensorSpec(
                                shape=(n_agents, 1)
                            )
                        },
                        shape=(n_agents,),
                    )
                }
            )

        if self.state_spec is not None:
            modules.append(
                TensorDictModule(
                    lambda state, action: torch.cat(
                        [state, action.view(action.shape[:-2], -1)], dim=-1
                    ),
                    in_keys=["state", (group, "action")],
                    out_keys=["state_action"],
                )
            )
            critic_input_spec = CompositeSpec(
                {
                    "state_action": UnboundedContinuousTensorSpec(
                        shape=(
                            self.state_spec["state"].shape[-1]
                            + self.action_spec[group, "action"].shape[-1] * n_agents,
                        )
                    )
                }
            )

            modules.append(
                self.model_config.get_model(
                    input_spec=critic_input_spec,
                    output_spec=critic_output_spec,
                    n_agents=n_agents,
                    centralised=True,
                    input_has_agent_dim=False,
                    agent_group=group,
                    share_params=self.share_param_critic,
                    device=self.device,
                )
            )

        else:

            modules.append(
                TensorDictModule(
                    lambda obs, action: torch.cat([obs, action], dim=-1),
                    in_keys=[(group, "observation"), (group, "action")],
                    out_keys=[(group, "obs_action")],
                )
            )
            critic_input_spec = CompositeSpec(
                {
                    group: CompositeSpec(
                        {
                            "obs_action": UnboundedContinuousTensorSpec(
                                shape=(
                                    n_agents,
                                    self.observation_spec[group, "observation"].shape[
                                        -1
                                    ]
                                    + self.action_spec[group, "action"].shape[-1],
                                )
                            )
                        },
                        shape=(n_agents,),
                    )
                }
            )

            modules.append(
                self.model_config.get_model(
                    input_spec=critic_input_spec,
                    output_spec=critic_output_spec,
                    n_agents=n_agents,
                    centralised=True,
                    input_has_agent_dim=True,
                    agent_group=group,
                    share_params=self.share_param_critic,
                    device=self.device,
                )
            )

        if self.share_param_critic:
            modules.append(
                TensorDictModule(
                    lambda value: value.unsqueeze(-2).expand(
                        *value.shape[:-1], n_agents, 1
                    ),
                    in_keys=["state_action_value"],
                    out_keys=[(group, "state_action_value")],
                )
            )

        return TensorDictSequential(*modules)


@dataclass
class MaddpgConfig(AlgorithmConfig):
    # You can add any kwargs from benchmarl.algorithms.Maddpg

    share_param_actor: bool = True
    share_param_critic: bool = True

    loss_function: str = "l2"
    delay_value: bool = True

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Maddpg
