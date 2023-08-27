from dataclasses import dataclass
from typing import Optional, Sequence, Type, Tuple

import torch
from torch import nn
from torchrl.modules import MultiAgentMLP


from benchmarl.models.common import ModelConfig, Model
from benchmarl.utils import DEVICE_TYPING


class Mlp(Model):
    def __init__(
        self,
        n_agents: int,
        input_features_shape: Tuple[int],
        output_features_shape: Tuple[int],
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        **kwargs,
    ):
        super().__init__(
            n_agents=n_agents,
            input_features_shape=input_features_shape,
            output_features_shape=output_features_shape,
            centralised=centralised,
            share_params=share_params,
            device=device,
        )
        if len(self.input_features_shape) != 1:
            raise ValueError(
                f"Input feature shape of MLP must be one dimensional, got {self.input_features_shape}"
            )
        if len(self.output_features_shape) != 1:
            raise ValueError(
                f"Output feature shape of MLP must be one dimensional, got {self.output_features_shape}"
            )

        self.mlp = MultiAgentMLP(
            n_agent_inputs=self.input_features_shape[0],
            n_agent_outputs=self.output_features_shape[0],
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            **kwargs,
        )

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        return self.mlp.forward(*inputs)


@dataclass
class MlpConfig(ModelConfig):

    # You can add any kwargs from torchrl.modules.MLP

    num_cells: Sequence[int] = (256, 256)
    layer_class: Type[nn.Module] = nn.Linear

    activation_class: Type[nn.Module] = nn.Tanh
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return Mlp


if __name__ == "__main__":
    print(
        MlpConfig().get_model(
            n_agents=3,
            input_features_shape=(30,),
            output_features_shape=(20,),
            centralised=True,
            share_params=False,
            device="cpu",
        )
    )
