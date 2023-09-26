import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    ReplayBuffer,
)
from torchrl.objectives import LossModule
from torchrl.objectives.utils import HardUpdate, SoftUpdate, TargetNetUpdater

from benchmarl.models.common import ModelConfig
from benchmarl.utils import DEVICE_TYPING, read_yaml_config


class Algorithm(ABC):
    def __init__(
        self,
        experiment_config: "DictConfig",  # noqa: F821
        model_config: ModelConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        state_spec: Optional[CompositeSpec],
        action_mask_spec: Optional[CompositeSpec],
        group_map: Dict[str, List[str]],
        on_policy: bool,
    ):
        self.device: DEVICE_TYPING = experiment_config.train_device

        self.experiment_config = experiment_config
        self.model_config = model_config
        self.on_policy = on_policy
        self.group_map = group_map
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.state_spec = state_spec
        self.action_mask_spec = action_mask_spec

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
        return self._get_replay_buffer(
            group=group,
            memory_size=self.experiment_config.replay_buffer_memory_size(
                self.on_policy
            ),
            sampling_size=self.experiment_config.train_minibatch_size(self.on_policy),
            traj_len=self.experiment_config.traj_len,
            storing_device=self.device,
        )

    def get_policy_for_loss(self, group: str) -> TensorDictModule:
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
        raise NotImplementedError

    @abstractmethod
    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        raise NotImplementedError

    @abstractmethod
    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        raise NotImplementedError

    @abstractmethod
    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        raise NotImplementedError

    @abstractmethod
    def _get_replay_buffer(
        self,
        group: str,
        memory_size: int,
        sampling_size: int,
        traj_len: int,
        storing_device: DEVICE_TYPING,
    ) -> ReplayBuffer:
        raise NotImplementedError

    @abstractmethod
    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        return loss_vals


@dataclass
class AlgorithmConfig:
    def get_algorithm(
        self,
        experiment_config,
        model_config: ModelConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        state_spec: CompositeSpec,
        action_mask_spec: Optional[CompositeSpec],
        group_map: Dict[str, List[str]],
    ) -> Algorithm:
        return self.associated_class()(
            **self.__dict__,
            experiment_config=experiment_config,
            model_config=model_config,
            observation_spec=observation_spec,
            action_spec=action_spec,
            state_spec=state_spec,
            action_mask_spec=action_mask_spec,
            group_map=group_map,
            on_policy=self.on_policy(),
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

    @staticmethod
    @abstractmethod
    def get_from_yaml(path: Optional[str] = None):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def associated_class() -> Type[Algorithm]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def on_policy() -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports_continuous_actions() -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports_discrete_actions() -> bool:
        raise NotImplementedError
