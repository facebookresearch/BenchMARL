#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING
from typing import List, Optional, Sequence, Tuple, Type, Union

import torch

from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import ConvNet, MLP, MultiAgentConvNet, MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig


def number_conv_outputs(
    n_conv_inputs: Union[int, Tuple[int, int]],
    paddings: List[Union[int, Tuple[int, int]]],
    kernel_sizes: List[Union[int, Tuple[int, int]]],
    strides: List[Union[int, Tuple[int, int]]],
) -> Tuple[int, int]:
    if not isinstance(n_conv_inputs, int):
        n_conv_inputs_x, n_conv_inputs_y = n_conv_inputs
    else:
        n_conv_inputs_x = n_conv_inputs_y = n_conv_inputs
    for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
        if not isinstance(kernel_size, int):
            kernel_size_x, kernel_size_y = kernel_size
        else:
            kernel_size_x = kernel_size_y = kernel_size
        if not isinstance(padding, int):
            padding_x, padding_y = padding
        else:
            padding_x = padding_y = padding
        if not isinstance(stride, int):
            stride_x, stride_y = stride
        else:
            stride_x = stride_y = stride

        n_conv_inputs_x = (
            n_conv_inputs_x + 2 * padding_x - kernel_size_x
        ) // stride_x + 1
        n_conv_inputs_y = (
            n_conv_inputs_y + 2 * padding_y - kernel_size_y
        ) // stride_y + 1

    return n_conv_inputs_x, n_conv_inputs_y


class Cnn(Model):
    """CNN model."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
        )

        self.x = self.input_leaf_spec.shape[-3]
        self.y = self.input_leaf_spec.shape[-2]
        self.input_features = self.input_leaf_spec.shape[-1]

        self.output_features = self.output_leaf_spec.shape[-1]

        mlp_net_kwargs = {
            "_".join(k.split("_")[1:]): v
            for k, v in kwargs.items()
            if k.startswith("mlp_")
        }
        cnn_net_kwargs = {
            "_".join(k.split("_")[1:]): v
            for k, v in kwargs.items()
            if k.startswith("cnn_")
        }

        if self.input_has_agent_dim:
            self.cnn = MultiAgentConvNet(
                in_features=self.input_features,
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                **cnn_net_kwargs,
            )
            example_net = self.cnn._empty_net
        else:
            self.cnn = nn.ModuleList(
                [
                    ConvNet(
                        in_features=self.input_features,
                        device=self.device,
                        **cnn_net_kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )
            example_net = self.cnn[0]

        out_features = example_net.out_features
        out_x, out_y = number_conv_outputs(
            n_conv_inputs=(self.x, self.y),
            kernel_sizes=example_net.kernel_sizes,
            paddings=example_net.paddings,
            strides=example_net.strides,
        )
        cnn_output_size = out_features * out_x * out_y

        if self.output_has_agent_dim:
            self.mlp = MultiAgentMLP(
                n_agent_inputs=cnn_output_size,
                n_agent_outputs=self.output_features,
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                **mlp_net_kwargs,
            )
        else:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_features=cnn_output_size,
                        out_features=self.output_features,
                        device=self.device,
                        **mlp_net_kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        input = tensordict.get(self.in_key)
        # BenchMARL images are X,Y,C -> we convert them to C, X, Y for processing in TorchRL models
        input = input.transpose(-3, -1).transpose(-2, -1)

        # Has multi-agent input dimension
        if self.input_has_agent_dim:
            cnn_out = self.cnn.forward(input)
            if not self.output_has_agent_dim:
                # If we are here the module is centralised and parameter shared.
                # Thus the multi-agent dimension has been expanded,
                # We remove it without loss of data
                cnn_out = cnn_out[..., 0, :]

        # Does not have multi-agent input dimension
        else:
            if not self.share_params:
                cnn_out = torch.stack(
                    [net(input) for net in self.cnn],
                    dim=-2,
                )
            else:
                cnn_out = self.cnn[0](input)

        # Cnn output has multi-agent input dimension
        if self.output_has_agent_dim:
            res = self.mlp.forward(cnn_out)
        else:
            if not self.share_params:
                res = torch.stack(
                    [net(cnn_out) for net in self.mlp],
                    dim=-2,
                )
            else:
                res = self.mlp[0](cnn_out)

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class CnnConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Cnn`."""

    cnn_num_cells: Sequence[int] = MISSING
    cnn_kernel_sizes: Sequence[int] = MISSING
    cnn_strides: Sequence[int] = MISSING
    cnn_paddings: Sequence[int] = MISSING
    cnn_activation_class: Type[nn.Module] = MISSING

    mlp_num_cells: Sequence[int] = MISSING
    mlp_layer_class: Type[nn.Module] = MISSING
    mlp_activation_class: Type[nn.Module] = MISSING

    cnn_activation_kwargs: Optional[dict] = None
    cnn_norm_class: Type[nn.Module] = None
    cnn_norm_kwargs: Optional[dict] = None

    mlp_activation_kwargs: Optional[dict] = None
    mlp_norm_class: Type[nn.Module] = None
    mlp_norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return Cnn
