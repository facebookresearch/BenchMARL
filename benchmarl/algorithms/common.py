#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pathlib

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import (
    Categorical,
    LazyMemmapStorage,
    LazyTensorStorage,
    OneHot,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import (
    PrioritizedSampler,
    RandomSampler,
    SamplerWithoutReplacement,
)
from torchrl.envs import Compose, EnvBase, Transform
from torchrl.objectives import LossModule
from torchrl.objectives.utils import HardUpdate, SoftUpdate, TargetNetUpdater

from benchmarl.models.common import ModelConfig
from benchmarl.utils import _read_yaml_config, DEVICE_TYPING


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
        self.buffer_device: DEVICE_TYPING = experiment.config.buffer_device
        self.experiment_config = experiment.config
        self.model_config = experiment.model_config
        self.critic_model_config = experiment.critic_model_config
        self.on_policy = experiment.on_policy
        self.group_map = experiment.group_map
        self.observation_spec = experiment.observation_spec
        self.action_spec = experiment.action_spec
        self.state_spec = experiment.state_spec
        self.action_mask_spec = experiment.action_mask_spec
        self.has_independent_critic = (
            experiment.algorithm_config.has_independent_critic()
        )
        self.has_centralized_critic = (
            experiment.algorithm_config.has_centralized_critic()
        )
        self.has_critic = experiment.algorithm_config.has_critic
        self.has_rnn = self.model_config.is_rnn or (
            self.critic_model_config.is_rnn and self.has_critic
        )

        # Cached values that will be instantiated only once and then remain fixed
        self._losses_and_updaters = {}
        self._policies_for_loss = {}
        self._policies_for_collection = {}

        self._check_specs()

    def _check_specs(self):
        if self.state_spec is not None:
            if len(self.state_spec.keys(True, True)) != 1:
                raise ValueError(
                    "State spec must contain one entry per group"
                    " to follow the library conventions, "
                    "you can apply a transform to your environment to satisfy this criteria."
                )
        for group in self.group_map.keys():
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
        This function calls the abstract :class:`~benchmarl.algorithms.Algorithm._get_loss()` which needs to be implemented.
        The function will cache the output at the first call and return the cached values in future calls.

        Args:
            group (str): agent group of the loss and updater

        Returns: LossModule and TargetNetUpdater for the group
        """
        if group not in self._losses_and_updaters.keys():
            action_space = self.action_spec[group, "action"]
            continuous = not isinstance(action_space, (Categorical, OneHot))
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
        self, group: str, transforms: List[Transform] = None
    ) -> ReplayBuffer:
        """
        Get the ReplayBuffer for a specific group.
        This function will check ``self.on_policy`` and create the buffer accordingly

        Args:
            group (str): agent group of the loss and updater
            transforms (optional, list of Transform): Transforms to apply to the replay buffer ``.sample()`` call

        Returns: ReplayBuffer the group
        """
        memory_size = self.experiment_config.replay_buffer_memory_size(self.on_policy)
        sampling_size = self.experiment_config.train_minibatch_size(self.on_policy)
        if self.has_rnn:
            sequence_length = -(
                -self.experiment_config.collected_frames_per_batch(self.on_policy)
                // self.experiment_config.n_envs_per_worker(self.on_policy)
            )
            memory_size = -(-memory_size // sequence_length)
            sampling_size = -(-sampling_size // sequence_length)

        # Sampler
        if self.on_policy:
            sampler = SamplerWithoutReplacement()
        elif self.experiment_config.off_policy_use_prioritized_replay_buffer:
            sampler = PrioritizedSampler(
                memory_size,
                self.experiment_config.off_policy_prb_alpha,
                self.experiment_config.off_policy_prb_beta,
            )
        else:
            sampler = RandomSampler()

        # Storage
        if self.buffer_device == "disk" and not self.on_policy:
            storage = LazyMemmapStorage(
                memory_size,
                device=self.device,
                scratch_dir=self.experiment.folder_name / f"buffer_{group}",
            )
        else:
            storage = LazyTensorStorage(
                memory_size,
                device=self.device if self.on_policy else self.buffer_device,
            )

        return TensorDictReplayBuffer(
            storage=storage,
            sampler=sampler,
            batch_size=sampling_size,
            priority_key=(group, "td_error"),
            transform=Compose(*transforms) if transforms is not None else None,
        )

    def get_policy_for_loss(self, group: str) -> TensorDictModule:
        """
        Get the non-explorative policy for a specific group loss.
        This function calls the abstract :class:`~benchmarl.algorithms.Algorithm._get_policy_for_loss()` which needs to be implemented.
        The function will cache the output at the first call and return the cached values in future calls.

        Args:
            group (str): agent group of the policy

        Returns: TensorDictModule representing the policy
        """
        if group not in self._policies_for_loss.keys():
            action_space = self.action_spec[group, "action"]
            continuous = not isinstance(action_space, (Categorical, OneHot))
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
        This function calls the abstract :class:`~benchmarl.algorithms.Algorithm._get_policy_for_collection()` which needs to be implemented.
        The function will cache the output at the first call and return the cached values in future calls.

        Returns: TensorDictSequential representing all explorative policies
        """
        policies = []
        for group in self.group_map.keys():
            if group not in self._policies_for_collection.keys():
                policy_for_loss = self.get_policy_for_loss(group)
                action_space = self.action_spec[group, "action"]
                continuous = not isinstance(action_space, (Categorical, OneHot))
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
        This function calls the abstract :class:`~benchmarl.algorithms.Algorithm._get_parameters()` which needs to be implemented.

        Returns: a dictionary mapping loss names to a parameters' list
        """
        return self._get_parameters(
            group=group,
            loss=self.get_loss_and_updater(group)[0],
        )

    def process_env_fun(
        self,
        env_fun: Callable[[], EnvBase],
    ) -> Callable[[], EnvBase]:
        """
        This function can be used to wrap env_fun

        Args:
            env_fun (callable): a function that takes no args and creates an enviornment

        Returns: a function that takes no args and creates an enviornment

        """

        return env_fun

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
        return _read_yaml_config(str(yaml_path.resolve()))

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        """
        Load the algorithm configuration from yaml

        Args:
            path (str, optional): The full path of the yaml file to load from.
                If None, it will default to
                ``benchmarl/conf/algorithm/self.associated_class().__name__``

        Returns: the loaded AlgorithmConfig
        """

        if path is None:
            config = AlgorithmConfig._load_from_yaml(
                name=cls.associated_class().__name__
            )

        else:
            config = _read_yaml_config(path)
        return cls(**config)

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

    @staticmethod
    def has_independent_critic() -> bool:
        """
        If the algorithm uses an independent critic
        """
        return False

    @staticmethod
    def has_centralized_critic() -> bool:
        """
        If the algorithm uses a centralized critic
        """
        return False

    def has_critic(self) -> bool:
        """
        If the algorithm uses a critic
        """
        if self.has_centralized_critic() and self.has_independent_critic():
            raise ValueError(
                "Algorithm can either have a centralized critic or an indpendent one"
            )
        return self.has_centralized_critic() or self.has_independent_critic()
