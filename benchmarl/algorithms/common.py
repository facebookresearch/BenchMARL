from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Type

from benchmarl.models.common import ModelConfig, Model
from tensordict.nn import TensorDictModule
from torchrl import ReplayBuffer
from torchrl import LossModule
from torchrl.data import CompositeSpec
from benchmarl.utils import DEVICE_TYPING


class Algorithm(ABC):
    def __init__(
        self,
        model_config: ModelConfig,
        n_agents: int,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
    ):
        self.model_config = model_config

        self.observation_spec = observation_spec
        self.action_spec = action_spec

    @abstractmethod
    def get_policy(self) -> TensorDictModule:
        raise NotImplementedError

    @abstractmethod
    def get_replay_buffer(
        self,
        memory_size: int,
        sampling_size: int,
        storing_device: DEVICE_TYPING,
    ) -> ReplayBuffer:
        raise NotImplementedError

    @abstractmethod
    def get_loss(self) -> LossModule:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports_continuous_actions() -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports_discrete_actions() -> bool:
        raise NotImplementedError


@dataclass
class AlgorithmConfig(ABC):
    def get_algorithm(
        self,
        model_config: ModelConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
    ) -> Algorithm:
        return self.associated_class()(
            **asdict(self),
            model_config=model_config,
            observation_spec=observation_spec,
            action_spec=action_spec
        )

    @staticmethod
    @abstractmethod
    def associated_class() -> Type[Algorithm]:
        raise NotImplementedError
