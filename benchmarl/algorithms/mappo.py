#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Tuple, Type

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.modules import MaskedCategorical, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig


class Mappo(Algorithm):
    def __init__(
        self,
        share_param_critic: bool,
        clip_epsilon: float,
        entropy_coef: bool,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.share_param_critic = share_param_critic
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.loss_critic_type = loss_critic_type
        self.lmbda = lmbda

    #############################
    # Overridden abstract methods
    #############################

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:

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
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        return loss_module, False

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:

        return {
            "loss_objective": list(loss.actor_params.flatten_keys().values()),
            "loss_critic": list(loss.critic_params.flatten_keys().values()),
        }

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
            share_params=self.experiment_config.share_policy_params,
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
                spec=self.action_spec[group, "action"],
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
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
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
        # MAPPO uses the same stochastic actor for collection
        return policy_for_loss

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
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
            "loss_objective", loss_vals["loss_objective"] + loss_vals["loss_entropy"]
        )
        del loss_vals["loss_entropy"]
        return loss_vals

    #####################
    # Custom new methods
    #####################

    def get_critic(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        if self.share_param_critic:
            critic_output_spec = CompositeSpec(
                {"state_value": UnboundedContinuousTensorSpec(shape=(1,))}
            )
        else:
            critic_output_spec = CompositeSpec(
                {
                    group: CompositeSpec(
                        {
                            "state_value": UnboundedContinuousTensorSpec(
                                shape=(n_agents, 1)
                            )
                        },
                        shape=(n_agents,),
                    )
                }
            )

        if self.state_spec is not None:
            value_module = self.critic_model_config.get_model(
                input_spec=self.state_spec,
                output_spec=critic_output_spec,
                n_agents=n_agents,
                centralised=True,
                input_has_agent_dim=False,
                agent_group=group,
                share_params=self.share_param_critic,
                device=self.device,
            )

        else:
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
            value_module = self.critic_model_config.get_model(
                input_spec=critic_input_spec,
                output_spec=critic_output_spec,
                n_agents=n_agents,
                centralised=True,
                input_has_agent_dim=True,
                agent_group=group,
                share_params=self.share_param_critic,
                device=self.device,
            )
        if self.share_param_critic:
            expand_module = TensorDictModule(
                lambda value: value.unsqueeze(-2).expand(
                    *value.shape[:-1], n_agents, 1
                ),
                in_keys=["state_value"],
                out_keys=[(group, "state_value")],
            )
            value_module = TensorDictSequential(value_module, expand_module)

        return value_module


@dataclass
class MappoConfig(AlgorithmConfig):

    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Mappo

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True
