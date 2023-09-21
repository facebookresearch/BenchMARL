from dataclasses import dataclass, MISSING
from typing import Dict, Optional, Tuple, Type

import torch
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
from torchrl.modules import EGreedyModule, QValueModule
from torchrl.objectives import ClipPPOLoss, DQNLoss, LossModule, ValueEstimators
from torchrl.objectives.utils import SoftUpdate, TargetNetUpdater

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig
from benchmarl.utils import DEVICE_TYPING, read_yaml_config


class Iql(Algorithm):
    def __init__(self, delay_value: bool, loss_function: str, **kwargs):
        super().__init__(**kwargs)

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
            raise NotImplementedError("Iql is not compatible with continuous actions.")
        else:
            # Loss
            loss_module = DQNLoss(
                policy_for_loss,
                delay_value=self.delay_value,
                loss_function=self.loss_function,
                action_space=self.action_spec,
            )
            loss_module.set_keys(
                reward=(group, "reward"),
                action=(group, "action"),
                done=(group, "done"),
                action_value=(group, "action_value"),
                value=(group, "chosen_action_value"),
                priority=(group, "td_error"),
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
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
        )
        if self.action_mask_spec is not None:
            action_mask_key = (group, "action_mask")
        else:
            action_mask_key = None

        value_module = QValueModule(
            action_value_key=(group, "action_value"),
            action_mask_key=action_mask_key,
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
        if self.action_mask_spec is not None:
            action_mask_key = (group, "action_mask")
        else:
            action_mask_key = None

        greedy = EGreedyModule(
            annealing_num_steps=self.experiment_config.exploration_annealing_num_frames,
            action_key=(group, "action"),
            spec=self.action_spec[(group, "action")],
            action_mask_key=action_mask_key,
            # eps_init = 1.0,
            # eps_end = 0.1,
        )
        return TensorDictSequential(*policy_for_loss, greedy)

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

    #####################
    # Custom new methods
    #####################


@dataclass
class IqlConfig(AlgorithmConfig):

    delay_value: bool = MISSING
    loss_function: str = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Iql

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False

    @staticmethod
    def get_from_yaml(path: Optional[str] = None):
        if path is None:
            return IqlConfig(
                **AlgorithmConfig._load_from_yaml(
                    name=IqlConfig.associated_class().__name__,
                )
            )
        else:
            return IqlConfig(**read_yaml_config(path))
