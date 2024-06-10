#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Tuple, Type

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.modules import EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, LossModule, ValueEstimators


class CustomAlgorithm(Algorithm):
    def __init__(
        self, delay_value: bool, loss_function: str, my_custom_arg: int, **kwargs
    ):
        # In the init function you can define the init parameters you need, just make sure
        # to pass the kwargs to the super() class
        super().__init__(**kwargs)

        self.delay_value = delay_value
        self.loss_function = loss_function
        self.my_custom_arg = my_custom_arg

        # In all the class you have access to a lot of extra things like
        self.my_custom_method()  # Custom methods
        _ = self.experiment  # Experiment class
        _ = self.experiment_config  # Experiment config
        _ = self.model_config  # Policy config
        _ = self.critic_model_config  # Eventual critic config
        _ = self.group_map  # The group to agent names map

        # Specs
        _ = self.observation_spec
        _ = self.action_spec
        _ = self.state_spec
        _ = self.action_mask_spec

    #############################
    # Overridden abstract methods
    #############################

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if continuous:
            raise NotImplementedError(
                "Custom Iql is not compatible with continuous actions."
            )
        else:
            # Loss
            loss_module = DQNLoss(
                policy_for_loss,
                delay_value=self.delay_value,
                loss_function=self.loss_function,
                action_space=self.action_spec[group, "action"],
            )
            # Always tell the loss where to finc the data
            # You can make sure the data is in the right place in self.process_batch
            # This loss for example expects all data to have the multagent dimension so we take care of that in
            # self.process_batch
            loss_module.set_keys(
                reward=(group, "reward"),
                action=(group, "action"),
                done=(group, "done"),
                terminated=(group, "terminated"),
                action_value=(group, "action_value"),
                value=(group, "chosen_action_value"),
                priority=(group, "td_error"),
            )
            # Choose your value estimator, see what is available in the ValueEstimators enum
            loss_module.make_value_estimator(
                ValueEstimators.TD0, gamma=self.experiment_config.gamma
            )
            # This loss has target delayed parameters so the second value is True
            return loss_module, True

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        # For each loss name, associate it the parameters you want
        # You can optionally modify (aggregate) loss names in self.process_loss_vals()
        return {"loss": loss.parameters()}

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        if continuous:
            raise ValueError("This should never happen")

        # The number of agents in the group
        n_agents = len(self.group_map[group])
        # The shape of the discrete action
        logits_shape = [
            *self.action_spec[group, "action"].shape,
            self.action_spec[group, "action"].space.n,
        ]

        # This is the spec of the policy input for this group
        actor_input_spec = CompositeSpec(
            {group: self.observation_spec[group].clone().to(self.device)}
        )
        # This is the spec of the policy output for this group
        actor_output_spec = CompositeSpec(
            {
                group: CompositeSpec(
                    {"action_value": UnboundedContinuousTensorSpec(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )
        # This is our neural policy
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,  # Always true for a policy
            n_agents=n_agents,
            centralised=False,  # Always false for a policy
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
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
            action_space=None,  # We already passed the spec
        )

        # Here we chain the actor and the value module to get our policy
        return TensorDictSequential(actor_module, value_module)

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        if self.action_mask_spec is not None:
            action_mask_key = (group, "action_mask")
        else:
            action_mask_key = None
        # Add exploration for collection
        greedy = EGreedyModule(
            annealing_num_steps=self.experiment_config.get_exploration_anneal_frames(
                self.on_policy
            ),
            action_key=(group, "action"),
            spec=self.action_spec[(group, "action")],
            action_mask_key=action_mask_key,
            eps_init=self.experiment_config.exploration_eps_init,
            eps_end=self.experiment_config.exploration_eps_end,
        )
        return TensorDictSequential(*policy_for_loss, greedy)

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        # Here we make sure that all entries have the desired shape,
        # thus, in case there are shared dones, terminated, or rewards, we expande them

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

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        # Here you can modify the loss_vals tensordict containing entries loss_name->loss_value
        # For example you can sum two entries in a new entry to optimize them together.
        return loss_vals

    #####################
    # Custom new methods
    #####################
    def my_custom_method(self):
        pass


@dataclass
class CustomAlgorithmConfig(AlgorithmConfig):
    # This is a class representing the configuration of your algorithm
    # It will be used to validate loaded configs, so that everytime you load this algorithm
    # we know exactly which and what parameters to expect with their types

    # This is a list of args passed to your algorithm
    delay_value: bool = MISSING
    loss_function: str = MISSING
    my_custom_arg: int = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        # The associated algorithm class
        return CustomAlgorithm

    @staticmethod
    def supports_continuous_actions() -> bool:
        # Is it compatible with continuous actions?
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        # Is it compatible with discrete actions?
        return True

    @staticmethod
    def on_policy() -> bool:
        # Should it be trained on or off policy?
        return False
