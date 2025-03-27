#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Optional, Tuple, Type, Union

from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import (
    IndependentNormal,
    MaskedCategorical,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import DiscreteSACLoss, LossModule, SACLoss, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig


class Isac(Algorithm):
    """Independent Soft Actor Critic.

    Args:
        share_param_critic (bool): Whether to share the parameters of the critics withing agent groups
        num_qvalue_nets (integer): number of Q-Value networks used.
        loss_function (str): loss function to be used with
            the value function loss.
        delay_qvalue (bool): Whether to separate the target Q value
            networks from the Q value networks used for data collection.
        target_entropy (float or str, optional): Target entropy for the
            stochastic policy. Default is "auto", where target entropy is
            computed as :obj:`-prod(n_actions)`.
        discrete_target_entropy_weight (float): weight for the target entropy term when actions are discrete
        alpha_init (float): initial entropy multiplier.
        min_alpha (float): min value of alpha.
        max_alpha (float): max value of alpha.
        fixed_alpha (bool): if ``True``, alpha will be fixed to its
            initial value. Otherwise, alpha will be optimized to
            match the 'target_entropy' value.
        scale_mapping (str): positive mapping function to be used with the std.
            choices: "softplus", "exp", "relu", "biased_softplus_1";
        use_tanh_normal (bool): if ``True``, use TanhNormal as the continuyous action distribution with support bound
            to the action domain. Otherwise, an IndependentNormal is used.

    """

    def __init__(
        self,
        share_param_critic: bool,
        num_qvalue_nets: int,
        loss_function: str,
        delay_qvalue: bool,
        target_entropy: Union[float, str],
        discrete_target_entropy_weight: float,
        alpha_init: float,
        min_alpha: Optional[float],
        max_alpha: Optional[float],
        fixed_alpha: bool,
        scale_mapping: str,
        use_tanh_normal: bool,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.share_param_critic = share_param_critic
        self.delay_qvalue = delay_qvalue
        self.num_qvalue_nets = num_qvalue_nets
        self.loss_function = loss_function
        self.target_entropy = target_entropy
        self.discrete_target_entropy_weight = discrete_target_entropy_weight
        self.alpha_init = alpha_init
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.fixed_alpha = fixed_alpha
        self.scale_mapping = scale_mapping
        self.use_tanh_normal = use_tanh_normal

    #############################
    # Overridden abstract methods
    #############################

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if continuous:
            # Loss
            loss_module = SACLoss(
                actor_network=policy_for_loss,
                qvalue_network=self.get_continuous_value_module(group),
                num_qvalue_nets=self.num_qvalue_nets,
                loss_function=self.loss_function,
                alpha_init=self.alpha_init,
                min_alpha=self.min_alpha,
                max_alpha=self.max_alpha,
                action_spec=self.action_spec,
                fixed_alpha=self.fixed_alpha,
                target_entropy=self.target_entropy,
                delay_qvalue=self.delay_qvalue,
            )
            loss_module.set_keys(
                state_action_value=(group, "state_action_value"),
                action=(group, "action"),
                reward=(group, "reward"),
                priority=(group, "td_error"),
                done=(group, "done"),
                terminated=(group, "terminated"),
            )

        else:
            loss_module = DiscreteSACLoss(
                actor_network=policy_for_loss,
                qvalue_network=self.get_discrete_value_module(group),
                num_qvalue_nets=self.num_qvalue_nets,
                loss_function=self.loss_function,
                alpha_init=self.alpha_init,
                min_alpha=self.min_alpha,
                max_alpha=self.max_alpha,
                action_space=self.action_spec,
                fixed_alpha=self.fixed_alpha,
                target_entropy=self.target_entropy,
                target_entropy_weight=self.discrete_target_entropy_weight,
                delay_qvalue=self.delay_qvalue,
                num_actions=self.action_spec[group, "action"].space.n,
            )
            loss_module.set_keys(
                action_value=(group, "action_value"),
                action=(group, "action"),
                reward=(group, "reward"),
                priority=(group, "td_error"),
                done=(group, "done"),
                terminated=(group, "terminated"),
            )

        loss_module.make_value_estimator(
            ValueEstimators.TD0, gamma=self.experiment_config.gamma
        )
        return loss_module, True

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        items = {
            "loss_actor": list(loss.actor_network_params.flatten_keys().values()),
            "loss_qvalue": list(loss.qvalue_network_params.flatten_keys().values()),
        }
        if not self.fixed_alpha:
            items.update({"loss_alpha": [loss.log_alpha]})
        return items

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

        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )

        actor_output_spec = Composite(
            {
                group: Composite(
                    {"logits": Unbounded(shape=logits_shape)},
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
            action_spec=self.action_spec,
        )

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=(
                    IndependentNormal if not self.use_tanh_normal else TanhNormal
                ),
                distribution_kwargs=(
                    {
                        "low": self.action_spec[(group, "action")].space.low,
                        "high": self.action_spec[(group, "action")].space.high,
                    }
                    if self.use_tanh_normal
                    else {}
                ),
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
                    distribution_kwargs={"neg_inf": -18.0},
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
        return policy

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
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

        return batch

    #####################
    # Custom new methods
    #####################

    def get_discrete_value_module(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        n_actions = self.action_spec[group, "action"].space.n

        critic_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )

        critic_output_spec = Composite(
            {
                group: Composite(
                    {"action_value": Unbounded(shape=(n_agents, n_actions))},
                    shape=(n_agents,),
                )
            }
        )
        value_module = self.critic_model_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=False,
            input_has_agent_dim=True,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
            action_spec=self.action_spec,
        )

        return value_module

    def get_continuous_value_module(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        modules = []

        critic_input_spec = Composite(
            {
                group: self.observation_spec[group]
                .clone()
                .update(self.action_spec[group])
            }
        )

        critic_output_spec = Composite(
            {
                group: Composite(
                    {"state_action_value": Unbounded(shape=(n_agents, 1))},
                    shape=(n_agents,),
                )
            }
        )

        modules.append(
            self.critic_model_config.get_model(
                input_spec=critic_input_spec,
                output_spec=critic_output_spec,
                n_agents=n_agents,
                centralised=False,
                input_has_agent_dim=True,
                agent_group=group,
                share_params=self.share_param_critic,
                device=self.device,
                action_spec=self.action_spec,
            )
        )

        return TensorDictSequential(*modules)


@dataclass
class IsacConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Isac`."""

    share_param_critic: bool = MISSING

    num_qvalue_nets: int = MISSING
    loss_function: str = MISSING
    delay_qvalue: bool = MISSING
    target_entropy: Union[float, str] = MISSING
    discrete_target_entropy_weight: float = MISSING

    alpha_init: float = MISSING
    min_alpha: Optional[float] = MISSING
    max_alpha: Optional[float] = MISSING
    fixed_alpha: bool = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Isac

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False

    @staticmethod
    def has_independent_critic() -> bool:
        return True
