from dataclasses import dataclass, MISSING
from typing import Dict, Optional, Tuple, Type

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import (
    CompositeSpec,
    ReplayBuffer,
    TensorDictReplayBuffer,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators
from torchrl.objectives.utils import TargetNetUpdater

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig
from benchmarl.utils import DEVICE_TYPING, read_yaml_config


class Ippo(Algorithm):
    def __init__(
        self,
        share_param_actor: bool,
        share_param_critic: bool,
        clip_epsilon: float,
        entropy_coef: bool,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.share_param_actor = share_param_actor
        self.share_param_critic = share_param_critic
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.loss_critic_type = loss_critic_type
        self.lmbda = lmbda

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
            sampler=SamplerWithoutReplacement(),
            batch_size=sampling_size,
        )

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, TargetNetUpdater]:

        # Loss
        loss_module = ClipPPOLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        return loss_module, None

    def _get_optimizers(
        self, group: str, loss: ClipPPOLoss, lr: float
    ) -> Dict[str, torch.optim.Optimizer]:

        return {"loss": torch.optim.Adam(loss.parameters(), lr=lr)}

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:

        n_agents = len(self.group_map[group])
        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
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
                    {"logits": UnboundedContinuousTensorSpec(shape=logits_shape)},
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

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec,
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=TanhNormal,
                distribution_kwargs={
                    "min": self.action_spec[(group, "action")].space.low,
                    "max": self.action_spec[(group, "action")].space.high,
                },
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )

        else:
            if self.action_mask_spec is None:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec,
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec,
                    in_keys={
                        "logits": (group, "logits"),
                        "mask": (group, "action_mask"),
                    },
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )

        return policy

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        # IPPO uses the same stochastic actor for collection
        return policy_for_loss

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

        with torch.no_grad():
            loss = self.get_loss_and_updater(group)[0]
            loss.value_estimator(
                batch,
                params=loss.critic_params,
                target_params=loss.target_critic_params,
            )

        return batch

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        loss_vals.set(
            "loss",
            loss_vals["loss_objective"]
            + loss_vals["loss_entropy"]
            + loss_vals["loss_critic"],
        )
        del loss_vals["loss_entropy"]
        del loss_vals["loss_objective"]
        del loss_vals["loss_critic"]
        return loss_vals

    #####################
    # Custom new methods
    #####################

    def get_critic(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])

        critic_input_spec = CompositeSpec(
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
        critic_output_spec = CompositeSpec(
            {
                group: CompositeSpec(
                    {"state_value": UnboundedContinuousTensorSpec(shape=(n_agents, 1))},
                    shape=(n_agents,),
                )
            }
        )
        value_module = self.model_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=False,
            input_has_agent_dim=True,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
        )

        return value_module


@dataclass
class IppoConfig(AlgorithmConfig):

    share_param_actor: bool = MISSING
    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Ippo

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def get_from_yaml(path: Optional[str] = None):
        if path is None:
            return IppoConfig(
                **AlgorithmConfig._load_from_yaml(
                    name=IppoConfig.associated_class().__name__,
                )
            )
        else:
            return IppoConfig(**read_yaml_config(path))
