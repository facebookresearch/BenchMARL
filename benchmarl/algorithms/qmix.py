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
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import EGreedyWrapper, QMixer, QValueModule
from torchrl.objectives import ClipPPOLoss, LossModule, QMixerLoss, ValueEstimators
from torchrl.objectives.utils import SoftUpdate, TargetNetUpdater

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig
from benchmarl.utils import DEVICE_TYPING


class Qmix(Algorithm):
    def __init__(
        self,
        mixing_embed_dim: int,
        delay_value: bool,
        loss_function: str,
        share_params: bool,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.delay_value = delay_value
        self.loss_function = loss_function
        self.mixing_embed_dim = mixing_embed_dim
        self.share_params = share_params

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
            priority_key="td_error",
        )

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, TargetNetUpdater]:
        if continuous:
            raise NotImplementedError("QMIX is not compatible with continuous actions.")
        else:
            # Loss
            loss_module = QMixerLoss(
                policy_for_loss,
                self.get_mixer(group),
                delay_value=self.delay_value,
                loss_function=self.loss_function,
                action_space=self.action_spec,
            )
            loss_module.set_keys(
                reward="reward",
                action=(group, "action"),
                done="done",
                action_value=(group, "action_value"),
                local_value=(group, "chosen_action_value"),
                global_value="chosen_action_value",
                priority="td_error",
            )
            loss_module.make_value_estimator(
                ValueEstimators.TD0, gamma=self.experiment_config.gamma
            )
            target_net_updater = SoftUpdate(
                loss_module, tau=self.experiment_config.polyak_tau
            )
            return loss_module, target_net_updater

    def _get_optimizers(
        self, group: str, loss: ClipPPOLoss, lr: float
    ) -> Dict[str, torch.optim.Optimizer]:

        return {
            "loss": torch.optim.Adam(loss.parameters(), lr=lr),
        }

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:

        n_agents = len(self.group_map[group])
        logits_shape = [
            *self.action_spec[group, "action"].shape,
            self.action_spec[group, "action"].space.n,
        ]

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
                    {"action_value": UnboundedContinuousTensorSpec(shape=logits_shape)},
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
            share_params=self.share_params,
            device=self.device,
        )
        if self.action_mask_spec is not None:
            raise NotImplementedError(
                "action mask is not yet compatible with q value modules"
            )
        value_module = QValueModule(
            action_value_key=(group, "action_value"),
            out_keys=[
                (group, "action"),
                (group, "action_value"),
                (group, "chosen_action_value"),
            ],
            spec=self.action_spec,
            action_space=None,
        )

        return TensorDictSequential(actor_module, value_module)

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:

        return EGreedyWrapper(
            policy_for_loss,
            annealing_num_steps=self.experiment_config.exploration_annealing_num_frames,
            action_key=(group, "action"),
            spec=self.action_spec[(group, "action")],
            # eps_init = 1.0,
            # eps_end = 0.1,
        )

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))

        done_key = ("next", "done")
        reward_key = ("next", "reward")

        if done_key not in keys:
            batch.set(
                done_key,
                batch.get(("next", group, "done")).mean(-2),
            )

        if reward_key not in keys:
            batch.set(
                reward_key,
                batch.get(("next", group, "reward")).mean(-2),
            )

        return batch

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False

    #####################
    # Custom new methods
    #####################

    def get_mixer(self, group: str) -> TensorDictModule:

        n_agents = len(self.group_map[group])

        if self.state_spec is not None:
            state_shape = self.state_spec["state"].shape
            in_keys = [(group, "chosen_action_value"), "state"]
        else:
            state_shape = self.observation_spec[group, "observation"].shape
            in_keys = [(group, "chosen_action_value"), (group, "observation")]

        mixer = TensorDictModule(
            module=QMixer(
                state_shape=state_shape,
                mixing_embed_dim=self.mixing_embed_dim,
                n_agents=n_agents,
                device=self.device,
            ),
            in_keys=in_keys,
            out_keys=["chosen_action_value"],
        )

        return mixer


@dataclass
class QmixConfig(AlgorithmConfig):
    # You can add any kwargs from benchmarl.algorithms.Qmix

    mixing_embed_dim: int = 32
    delay_value: bool = True
    loss_function: str = "l2"
    share_params: bool = True

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Qmix
