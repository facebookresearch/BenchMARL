#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Type

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import (
    DiscreteTensorSpec,
    LazyTensorStorage,
    OneHotDiscreteTensorSpec,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import RandomSampler, SamplerWithoutReplacement
from torchrl.objectives import LossModule
from torchrl.objectives.utils import HardUpdate, SoftUpdate, TargetNetUpdater

from benchmarl.models.common import ModelConfig
from benchmarl.utils import DEVICE_TYPING, read_yaml_config


class Algorithm(ABC):
    """
    Abstract class for an algorithm.
    This should be overridden by implemented algorithms
    and all abstract methods should be implemented.

     Args:
        experiment (Experiment): the experiment class
    """

    def __init__(self, experiment):
        self.experiment = experiment

        self.device: DEVICE_TYPING = experiment.config.train_device
        self.experiment_config = experiment.config
        self.model_config = experiment.model_config
        self.critic_model_config = experiment.critic_model_config
        self.on_policy = experiment.on_policy
        self.group_map = experiment.group_map
        self.observation_spec = experiment.observation_spec
        self.action_spec = experiment.action_spec
        self.state_spec = experiment.state_spec
        self.action_mask_spec = experiment.action_mask_spec

        # Cached values that will be instantiated only once and then remain fixed
        self._losses_and_updaters = {}
        self._policies_for_loss = {}
        self._policies_for_collection = {}

        self._check_specs()

    def _check_specs(self):
        if self.state_spec is not None:
            if (
                len(self.state_spec.keys(True, True)) != 1
                or list(self.state_spec.keys())[0] != "state"
            ):
                raise ValueError(
                    "State spec must contain one entry per group named 'state'"
                    " to follow the library conventions, "
                    "you can apply a transform to your environment to satisfy this criteria."
                )
        for group in self.group_map.keys():
            if (
                len(self.observation_spec[group].keys(True, True)) != 1
                or list(self.observation_spec[group].keys())[0] != "observation"
            ):
                raise ValueError(
                    "Observation spec must contain one entry per group named 'observation'"
                    " to follow the library conventions, "
                    "you can apply a transform to your environment to satisfy this criteria."
                )
            if (
                len(self.action_spec[group].keys(True, True)) != 1
                or list(self.action_spec[group].keys())[0] != "action"
            ):
                raise ValueError(
                    "Action spec must contain one entry per group named 'action'"
                    " to follow the library conventions, "
                    "you can apply a transform to your environment to satisfy this criteria."
                )
            if (
                self.action_mask_spec is not None
                and group in self.action_mask_spec.keys()
                and (
                    len(self.action_mask_spec[group].keys(True, True)) != 1
                    or list(self.action_mask_spec[group].keys())[0] != "action_mask"
                )
            ):
                raise ValueError(
                    "Action mask spec must contain one entry per group named 'action_mask'"
                    " to follow the library conventions, "
                    "you can apply a transform to your environment to satisfy this criteria."
                )

    def get_loss_and_updater(self, group: str) -> Tuple[LossModule, TargetNetUpdater]:
        """
        Get the LossModule and TargetNetUpdater for a specific group.
        This function calls the abstract self._get_loss() which needs to be implemented.
        The function will cache the output at the first call and return the cached values in future calls.

        Args:
            group (str): agent group of the loss and updater

        Returns: LossModule and TargetNetUpdater for the group

        """
        if group not in self._losses_and_updaters.keys():
            action_space = self.action_spec[group, "action"]
            continuous = not isinstance(
                action_space, (DiscreteTensorSpec, OneHotDiscreteTensorSpec)
            )
            loss, use_target = self._get_loss(
                group=group,
                policy_for_loss=self.get_policy_for_loss(group),
                continuous=continuous,
            )
            if use_target:
                if self.experiment_config.soft_target_update:
                    target_net_updater = SoftUpdate(
                        loss, tau=self.experiment_config.polyak_tau
                    )
                else:
                    target_net_updater = HardUpdate(
                        loss,
                        value_network_update_interval=self.experiment_config.hard_target_update_frequency,
                    )
            else:
                target_net_updater = None
            self._losses_and_updaters.update({group: (loss, target_net_updater)})
        return self._losses_and_updaters[group]

    def get_replay_buffer(
        self,
        group: str,
    ) -> ReplayBuffer:
        """
        Get the ReplayBuffer for a specific group.
        This function will check self.on_policy and create the buffer accordingly

        Args:
            group (str): agent group of the loss and updater

        Returns: ReplayBuffer the group
        """
        memory_size = self.experiment_config.replay_buffer_memory_size(self.on_policy)
        sampling_size = self.experiment_config.train_minibatch_size(self.on_policy)
        storing_device = self.device
        sampler = SamplerWithoutReplacement() if self.on_policy else RandomSampler()

        return TensorDictReplayBuffer(
            storage=LazyTensorStorage(memory_size, device=storing_device),
            sampler=sampler,
            batch_size=sampling_size,
        )

    def get_policy_for_loss(self, group: str) -> TensorDictModule:
        """
        Get the non-explorative policy for a specific group loss.
        This function calls the abstract self._get_policy_for_loss() which needs to be implemented.
        The function will cache the output at the first call and return the cached values in future calls.

        Args:
            group (str): agent group of the policy

        Returns: TensorDictModule representing the policy
        """
        if group not in self._policies_for_loss.keys():
            action_space = self.action_spec[group, "action"]
            continuous = not isinstance(
                action_space, (DiscreteTensorSpec, OneHotDiscreteTensorSpec)
            )
            self._policies_for_loss.update(
                {
                    group: self._get_policy_for_loss(
                        group=group,
                        continuous=continuous,
                        model_config=self.model_config,
                    )
                }
            )
        return self._policies_for_loss[group]

    def get_policy_for_collection(self) -> TensorDictSequential:
        """
        Get the explorative policy for all groups together.
        This function calls the abstract self._get_policy_for_collection() which needs to be implemented.
        The function will cache the output at the first call and return the cached values in future calls.

        Returns: TensorDictSequential representing all explorative policies
        """
        policies = []
        for group in self.group_map.keys():
            if group not in self._policies_for_collection.keys():
                policy_for_loss = self.get_policy_for_loss(group)
                action_space = self.action_spec[group, "action"]
                continuous = not isinstance(
                    action_space, (DiscreteTensorSpec, OneHotDiscreteTensorSpec)
                )
                policy_for_collection = self._get_policy_for_collection(
                    policy_for_loss,
                    group,
                    continuous,
                )
                self._policies_for_collection.update({group: policy_for_collection})
            policies.append(self._policies_for_collection[group])
        return TensorDictSequential(*policies)

    def get_parameters(self, group: str) -> Dict[str, Iterable]:
        """
        Get the dictionary mapping loss names to the relative parameters to optimize for a given group.
        This function calls the abstract self._get_parameters() which needs to be implemented.

        Returns: a dictionary mapping loss names to a parameters' list
        """
        return self._get_parameters(
            group=group,
            loss=self.get_loss_and_updater(group)[0],
        )

    ###############################
    # Abstract methods to implement
    ###############################

    @abstractmethod
    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        """
        Implement this function to return the LossModule for a specific group.

        Args:
            group (str): agent group of the loss
            policy_for_loss (TensorDictModule): the policy to use in the loss
            continuous (bool): whether to return a loss for continuous or discrete actions

        Returns: LossModule and a bool representing if the loss should have target parameters
        """
        raise NotImplementedError

    @abstractmethod
    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        """
        Get the dictionary mapping loss names to the relative parameters to optimize for a given group loss.

        Returns: a dictionary mapping loss names to a parameters' list
        """
        raise NotImplementedError

    @abstractmethod
    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        """
        Get the non-explorative policy for a specific group.

        Args:
            group (str): agent group of the policy
            model_config (ModelConfig): model config class
            continuous (bool): whether the policy should be continuous or discrete

        Returns: TensorDictModule representing the policy
        """
        raise NotImplementedError

    @abstractmethod
    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        """
        Implement this function to add an explorative layer to the policy used in the loss.

        Args:
            policy_for_loss (TensorDictModule): the group policy used in the loss
            group (str): agent group
            continuous (bool): whether the policy is continuous or discrete

        Returns: TensorDictModule representing the explorative policy
        """
        raise NotImplementedError

    @abstractmethod
    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        """
        This function can be used to reshape data coming from collection before it is passed to the policy.

        Args:
            group (str): agent group
            batch (TensorDictBase): the batch of data coming from the collector

        Returns: the processed batch

        """
        raise NotImplementedError

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        """
        Here you can modify the loss_vals tensordict containing entries loss_name->loss_value
        For example, you can sum two entries in a new entry, to optimize them together.

        Args:
            group (str): agent group
            loss_vals (TensorDictBase): the tensordict returned by the loss forward method

        Returns: the processed loss_vals
        """
        return loss_vals


@dataclass
class AlgorithmConfig:
    """
    Dataclass representing an algorithm configuration.
    This should be overridden by implemented algorithms.
    Implementors should:
     1. add configuration parameters for their algorithm
     2. implement all abstract methods
    """

    def get_algorithm(self, experiment) -> Algorithm:
        """
        Main function to turn the config into the associated algorithm
        Args:
            experiment (Experiment): the experiment class

        Returns: the Algorithm

        """
        return self.associated_class()(
            **self.__dict__,  # Passes all the custom config parameters
            experiment=experiment,
        )

    @staticmethod
    def _load_from_yaml(name: str) -> Dict[str, Any]:
        yaml_path = (
            pathlib.Path(__file__).parent.parent
            / "conf"
            / "algorithm"
            / f"{name.lower()}.yaml"
        )
        return read_yaml_config(str(yaml_path.resolve()))

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        """
        Load the algorithm configuration from yaml

        Args:
            path (str, optional): The full path of the yaml file to load from.
                If None, it will default to
                benchmarl/conf/algorithm/self.associated_class().__name__

        Returns: the loaded AlgorithmConfig
        """
        if path is None:
            return cls(
                **AlgorithmConfig._load_from_yaml(
                    name=cls.associated_class().__name__,
                )
            )
        else:
            return cls(**read_yaml_config(path))

    @staticmethod
    @abstractmethod
    def associated_class() -> Type[Algorithm]:
        """
        The algorithm class associated to the config
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def on_policy() -> bool:
        """
        If the algorithm has to be run on policy or off policy
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports_continuous_actions() -> bool:
        """
        If the algorithm supports continuous actions
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports_discrete_actions() -> bool:
        """
        If the algorithm supports discrete actions
        """
        raise NotImplementedError
