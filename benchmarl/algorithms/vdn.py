from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Optional, Tuple, Type

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
from torchrl.modules import EGreedyModule, QValueModule, VDNMixer
from torchrl.objectives import ClipPPOLoss, LossModule, QMixerLoss, ValueEstimators
from torchrl.objectives.utils import SoftUpdate, TargetNetUpdater

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig
from benchmarl.utils import DEVICE_TYPING, read_yaml_config


class Vdn(Algorithm):
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
            priority_key="td_error",
        )

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, TargetNetUpdater]:
        if continuous:
            raise NotImplementedError("Vdn is not compatible with continuous actions.")
        else:
            # Loss
            loss_module = QMixerLoss(
                policy_for_loss,
                self.get_mixer(group),
                delay_value=self.delay_value,
                loss_function=self.loss_function,
                action_space=self.action_spec[group, "action"],
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

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:

        return {
            "loss": loss.parameters(),
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
            spec=self.action_spec[group, "action"],
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

        done_key = ("next", "done")
        reward_key = ("next", "reward")

        if done_key not in keys:
            batch.set(
                done_key,
                batch.get(("next", group, "done")).any(-2),
            )

        if reward_key not in keys:
            batch.set(
                reward_key,
                batch.get(("next", group, "reward")).mean(-2),
            )

        return batch

    #####################
    # Custom new methods
    #####################

    def get_mixer(self, group: str) -> TensorDictModule:

        n_agents = len(self.group_map[group])
        mixer = TensorDictModule(
            module=VDNMixer(
                n_agents=n_agents,
                device=self.device,
            ),
            in_keys=[(group, "chosen_action_value")],
            out_keys=["chosen_action_value"],
        )

        return mixer


@dataclass
class VdnConfig(AlgorithmConfig):

    delay_value: bool = MISSING
    loss_function: str = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Vdn

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
            return VdnConfig(
                **AlgorithmConfig._load_from_yaml(
                    name=VdnConfig.associated_class().__name__,
                )
            )
        else:
            return VdnConfig(**read_yaml_config(path))
