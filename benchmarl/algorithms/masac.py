#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import warnings
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Optional, Tuple, Type, Union

import torch
import torch.nn.functional
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from tensordict.utils import _unravel_key_to_tuple, unravel_key
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


class Masac(Algorithm):
    """Multi Agent Soft Actor Critic.

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
        coupled_discrete_values (bool): only relevant for discrete action spaces. if ``True``, the critic will predict
            n_agents x n_actions action values given the global state (or concatenation of agents' observations). if ``False``,
            the critic will predict n_actions values given the global state and the actions of the other agents. This
            is done for all agents in parallel. ``True`` is more theoretically sound and should be preferred. However,
            if the number of outputs gets too large, you may want to try ``False``.
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
        coupled_discrete_values: bool,
        **kwargs,
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
        self.coupled_discrete_values = coupled_discrete_values

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
            if self.coupled_discrete_values and not self.share_param_critic:
                warnings.warn(
                    "disabling share_param_critic in MASAC with discrete actions and coupled_discrete_values has not effect"
                    "as the critic is already able to predict different values for different agents."
                )
            loss_module = DiscreteSACLoss(
                actor_network=policy_for_loss,
                qvalue_network=self.get_discrete_value_module_decoupled(group)
                if not self.coupled_discrete_values
                else self.get_discrete_value_module_coupled(group),
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
                    out_keys=[(group, "action")],
                    distribution_kwargs={"neg_inf": -18.0},
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

    def get_discrete_value_module_coupled(self, group: str) -> TensorDictModule:
        # Predict n_agents x n_actions values having access to the global state
        # this is more theoretically sound but might have a lot of outputs, for large number of agents you
        # may want to use the decoupled version
        n_agents = len(self.group_map[group])
        n_actions = self.action_spec[group, "action"].space.n

        critic_output_spec = Composite(
            {"action_value": Unbounded(shape=(n_actions * n_agents,))},
            device=self.device,
        )

        if self.state_spec is not None:
            critic_input_spec = self.state_spec
            input_has_agent_dim = False
        else:
            critic_input_spec = Composite(
                {group: self.observation_spec[group].clone().to(self.device)}
            )
            input_has_agent_dim = True

        value_module = self.critic_model_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=True,
            input_has_agent_dim=input_has_agent_dim,
            agent_group=group,
            share_params=True,
            device=self.device,
            action_spec=self.action_spec,
        )

        expand_module = TensorDictModule(
            lambda value: value.reshape(*value.shape[:-1], n_agents, n_actions),
            in_keys=["action_value"],
            out_keys=[(group, "action_value")],
        )
        value_module = TensorDictSequential(value_module, expand_module)

        return value_module

    def get_discrete_value_module_decoupled(self, group: str) -> TensorDictModule:
        # Predict n_actions values having access to the global state and the actions of other agents,
        # do this for all agents in parallel
        n_agents = len(self.group_map[group])
        n_actions = self.action_spec[group, "action"].space.n
        modules = []

        critic_output_spec = Composite(
            {
                group: Composite(
                    {"action_value": Unbounded(shape=(n_agents, n_actions))},
                    shape=(n_agents,),
                )
            },
            device=self.device,
        )
        modules.append(
            TensorDictModule(
                lambda action: _others_actions(
                    action, n_actions=n_actions, n_agents=n_agents
                ),
                in_keys=[(group, "logits")],
                out_keys=[(group, "others_action")],
            )
        )
        critic_input_spec = Composite(
            {
                group: Composite(
                    {
                        "others_action": Unbounded(
                            shape=(n_agents, n_actions * (n_agents - 1))
                        )
                    },
                    shape=(n_agents,),
                ),
            },
            device=self.device,
        )

        if self.state_spec is not None:
            global_state_key = _unravel_key_to_tuple(
                list(self.state_spec.keys(True, True))[0]
            )
            new_global_state_key = list(global_state_key)
            new_global_state_key[-1] = new_global_state_key[-1] + "_expanded"
            new_global_state_key = tuple(new_global_state_key)
            modules.append(
                TensorDictModule(
                    lambda state: state.unsqueeze(
                        -len(self.state_spec[global_state_key].shape) - 1
                    ).expand(
                        *state.shape[: -len(self.state_spec[global_state_key].shape)],
                        n_agents,
                        *self.state_spec[global_state_key].shape,
                    ),
                    in_keys=[global_state_key],
                    out_keys=[unravel_key((group, new_global_state_key))],
                )
            )
            critic_input_spec[group].update(
                {
                    new_global_state_key: self.state_spec[global_state_key]
                    .clone()
                    .unsqueeze(0)
                    .expand(n_agents, *self.state_spec[global_state_key].shape)
                    .to(self.device)
                }
            )
        else:
            observation_keys = list(self.observation_spec.keys(True, True))

            def process_keys(*observation_values):
                return_values = []
                for key, value in zip(observation_keys, observation_values):
                    spec = self.observation_spec[key]
                    batch_size = value.shape[: -len(spec.shape)]
                    value = value.repeat(
                        *(1 for _ in range(len(batch_size))),
                        n_agents,
                        *(1 for _ in range(len(spec.shape[1:]))),
                    )
                    value = value.view(
                        *batch_size,
                        n_agents,
                        *spec.shape[1:-1],
                        spec.shape[-1] * n_agents,
                    )
                    return_values.append(value)
                return tuple(return_values)

            def process_key(key):
                key = list(_unravel_key_to_tuple(key))
                key[-1] = key[-1] + "_expanded"
                return tuple(key)

            modules.append(
                TensorDictModule(
                    process_keys,
                    in_keys=observation_keys,
                    out_keys=[process_key(key) for key in observation_keys],
                )
            )
            critic_input_spec[group].update(
                {
                    process_key(key): val.reshape(
                        *val.shape[1:-1], val.shape[-1] * n_agents
                    )
                    .unsqueeze(0)
                    .expand(n_agents, *val.shape[1:-1], val.shape[-1] * n_agents)
                    .to(self.device)
                    for key, val in self.observation_spec[group].items()
                }
            )

        modules.append(
            self.critic_model_config.get_model(
                input_spec=critic_input_spec,
                output_spec=critic_output_spec,
                n_agents=n_agents,
                centralised=False,  # We handle the centralization in the code above
                input_has_agent_dim=True,
                agent_group=group,
                share_params=self.share_param_critic,
                device=self.device,
                action_spec=self.action_spec,
            )
        )

        return TensorDictSequential(*modules)

    def get_continuous_value_module(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        modules = []

        if self.share_param_critic:
            critic_output_spec = Composite(
                {"state_action_value": Unbounded(shape=(1,))}
            )
        else:
            critic_output_spec = Composite(
                {
                    group: Composite(
                        {"state_action_value": Unbounded(shape=(n_agents, 1))},
                        shape=(n_agents,),
                    )
                }
            )

        if self.state_spec is not None:

            modules.append(
                TensorDictModule(
                    lambda action: action.reshape(*action.shape[:-2], -1),
                    in_keys=[(group, "action")],
                    out_keys=["global_action"],
                )
            )

            critic_input_spec = self.state_spec.clone().update(
                {
                    "global_action": Unbounded(
                        shape=(self.action_spec[group, "action"].shape[-1] * n_agents,)
                    )
                }
            )

            modules.append(
                self.critic_model_config.get_model(
                    input_spec=critic_input_spec,
                    output_spec=critic_output_spec,
                    n_agents=n_agents,
                    centralised=True,
                    input_has_agent_dim=False,
                    agent_group=group,
                    share_params=self.share_param_critic,
                    device=self.device,
                    action_spec=self.action_spec,
                )
            )

        else:
            critic_input_spec = Composite(
                {
                    group: self.observation_spec[group]
                    .clone()
                    .update(self.action_spec[group])
                }
            )

            modules.append(
                self.critic_model_config.get_model(
                    input_spec=critic_input_spec,
                    output_spec=critic_output_spec,
                    n_agents=n_agents,
                    centralised=True,
                    input_has_agent_dim=True,
                    agent_group=group,
                    share_params=self.share_param_critic,
                    device=self.device,
                    action_spec=self.action_spec,
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


def _others_actions(logits, n_actions, n_agents):
    actions = logits.argmax(dim=-1)

    # input shape ..., n_agents
    batch_size = actions.shape[:-1]
    actions = torch.nn.functional.one_hot(
        actions, num_classes=n_actions
    )  # ..., n_agents, n_actions
    actions = actions.repeat(
        *(1 for _ in range(len(batch_size))), n_agents, 1
    )  # ..., 2* n_agents, n_actions
    actions = actions.view(*batch_size, n_agents, n_agents, n_actions)
    indices = (
        torch.eye(n_agents, n_agents, device=actions.device, dtype=torch.bool)
        .unsqueeze(-1)
        .expand(n_agents, n_agents, n_actions)
    )
    while len(indices.shape) < len(actions.shape):
        indices = indices.unsqueeze(0)
    indices = indices.expand(actions.shape)
    actions = actions.masked_select(
        ~indices
    )  # shape ..., n_agents, n_agents-1, n_actions

    actions = actions.view(*batch_size, n_agents, (n_agents - 1) * n_actions)
    # out shape ..., n_agents, n_agents-1 * n_actions
    return actions.to(torch.float32)


@dataclass
class MasacConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Masac`."""

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
    coupled_discrete_values: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Masac

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
    def has_centralized_critic() -> bool:
        return True
