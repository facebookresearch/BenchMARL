from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Sequence

import torch
from dataclasses import asdict
from torch import nn

from benchmarl.utils import DEVICE_TYPING


class Model(nn.Module, ABC):
    def __init__(
        self,
        n_agents: int,
        input_features_shape: Tuple[int],
        output_features_shape: Tuple[int],
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        **kwargs
    ):
        nn.Module.__init__(self)

        self.n_agents = n_agents
        self.input_features_shape = input_features_shape
        self.output_features_shape = output_features_shape
        self.centralised = centralised
        self.share_params = share_params
        self.device = device


class SequenceModel(Model):
    def __init__(
        self,
        models: Sequence[Model],
    ):
        for model in models:
            if model.n_agents != models[0].n_agents:
                raise ValueError(
                    "n_agents hase different values for models in the sequence"
                )
            if model.centralised != models[0].centralised:
                raise ValueError(
                    "centralised hase different values for models in the sequence"
                )
            if model.share_params != models[0].share_params:
                raise ValueError(
                    "share_params hase different values for models in the sequence"
                )
            if model.device != models[0].device:
                raise ValueError(
                    "n_agents hase different values for models in the sequence"
                )

        super().__init__(
            n_agents=models[0].n_agents,
            input_features_shape=models[0].input_features_shape,
            output_features_shape=models[-1].output_features_shape,
            centralised=models[0].centralised,
            share_params=models[0].share_params,
            device=models[0].device,
        )
        self.sequence = nn.Sequential(*models)

    def forward(self, *inputs):
        return self.sequence.forward(*inputs)


@dataclass
class ModelConfig(ABC):
    def get_model(
        self,
        n_agents: int,
        input_features_shape: Tuple[int],
        output_features_shape: Tuple[int],
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
    ) -> Model:
        return self.associated_class()(
            **asdict(self),
            n_agents=n_agents,
            input_features_shape=input_features_shape,
            output_features_shape=output_features_shape,
            centralised=centralised,
            share_params=share_params,
            device=device,
        )

    @staticmethod
    @abstractmethod
    def associated_class():
        raise NotImplementedError


class SequenceModelConfig(ModelConfig):
    def __init__(self, model_configs: Sequence[ModelConfig]):
        self.model_configs = model_configs

    def get_model(self, **kwargs) -> Model:
        return SequenceModel(
            [model_config.get_model(**kwargs) for model_config in self.model_configs]
        )

    @staticmethod
    def associated_class():
        raise NotImplementedError
