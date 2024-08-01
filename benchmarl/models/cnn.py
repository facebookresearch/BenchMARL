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


def _number_conv_outputs(
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
    """Convolutional Neural Network (CNN) model.

    The BenchMARL CNN accepts multiple inputs of 2 types:

    - images: Tensors of shape ``(*batch,X,Y,C)``
    - arrays: Tensors of shape ``(*batch,F)``

    The CNN model will check that all image inputs have the same shape (excluding the last dimension)
    and cat them along that dimension before processing them with :class:`torchrl.modules.ConvNet`.

    It will check that all array inputs have the same shape (excluding the last dimension)
    and cat them along that dimension.

    It will then cat the arrays and processed images and feed them to the MLP together.

    Args:
        cnn_num_cells (int or Sequence of int): number of cells of
            every layer in between the input and output. If an integer is
            provided, every layer will have the same number of cells. If an
            iterable is provided, the linear layers ``out_features`` will match
            the content of num_cells.
        cnn_kernel_sizes (int, sequence of int): Kernel size(s) of the
            conv network. If iterable, the length must match the depth,
            defined by the ``num_cells`` or depth arguments.
        cnn_strides (int or sequence of int): Stride(s) of the conv network. If
            iterable, the length must match the depth, defined by the
            ``num_cells`` or depth arguments.
        cnn_paddings: (int or Sequence of int): padding size for every layer.
        cnn_activation_class (Type[nn.Module] or callable): activation
            class or constructor to be used.
        cnn_activation_kwargs (dict or list of dicts, optional): kwargs to be used
            with the activation class. A list of kwargs of length ``depth``
            can also be passed, with one element per layer.
        cnn_norm_class (Type or callable, optional): normalization class or
            constructor, if any.
        cnn_norm_kwargs (dict or list of dicts, optional): kwargs to be used with
            the normalization layers. A list of kwargs of length ``depth`` can
            also be passed, with one element per layer.
        mlp_num_cells (int or Sequence[int]): number of cells of every layer in between the input and output. If
            an integer is provided, every layer will have the same number of cells. If an iterable is provided,
            the linear layers out_features will match the content of num_cells.
        mlp_layer_class (Type[nn.Module]): class to be used for the linear layers;
        mlp_activation_class (Type[nn.Module]): activation class to be used.
        mlp_activation_kwargs (dict, optional): kwargs to be used with the activation class;
        mlp_norm_class (Type, optional): normalization class, if any.
        mlp_norm_kwargs (dict, optional): kwargs to be used with the normalization layers;

    """

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
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        self.x = self.input_spec[self.image_in_keys[0]].shape[-3]
        self.y = self.input_spec[self.image_in_keys[0]].shape[-2]
        self.input_features_images = sum(
            [self.input_spec[key].shape[-1] for key in self.image_in_keys]
        )
        self.input_features_tensors = sum(
            [self.input_spec[key].shape[-1] for key in self.tensor_in_keys]
        )
        if self.input_has_agent_dim and not self.output_has_agent_dim:
            # In this case the tensor features will be centralized
            self.input_features_tensors *= self.n_agents

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
                in_features=self.input_features_images,
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
                        in_features=self.input_features_images,
                        device=self.device,
                        **cnn_net_kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )
            example_net = self.cnn[0]

        out_features = example_net.out_features
        out_x, out_y = _number_conv_outputs(
            n_conv_inputs=(self.x, self.y),
            kernel_sizes=example_net.kernel_sizes,
            paddings=example_net.paddings,
            strides=example_net.strides,
        )
        cnn_output_size = out_features * out_x * out_y

        if self.output_has_agent_dim:
            self.mlp = MultiAgentMLP(
                n_agent_inputs=cnn_output_size + self.input_features_tensors,
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
                        in_features=cnn_output_size + self.input_features_tensors,
                        out_features=self.output_features,
                        device=self.device,
                        **mlp_net_kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

    def _perform_checks(self):
        super()._perform_checks()

        input_shape_image = None
        self.image_in_keys = []
        input_shape_tensor = None
        self.tensor_in_keys = []
        for input_key, input_spec in self.input_spec.items(True, True):
            if (self.input_has_agent_dim and len(input_spec.shape) == 4) or (
                not self.input_has_agent_dim and len(input_spec.shape) == 3
            ):
                self.image_in_keys.append(input_key)
                if input_shape_image is None:
                    input_shape_image = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_image:
                    raise ValueError(
                        f"CNN image inputs should all have the same shape up to the last dimension, got {self.input_spec}"
                    )
            elif (self.input_has_agent_dim and len(input_spec.shape) == 2) or (
                not self.input_has_agent_dim and len(input_spec.shape) == 1
            ):
                self.tensor_in_keys.append(input_key)
                if input_shape_tensor is None:
                    input_shape_tensor = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_tensor:
                    raise ValueError(
                        f"CNN tensor inputs should all have the same shape up to the last dimension, got {self.input_spec}"
                    )
            else:
                raise ValueError(
                    f"CNN input value {input_key} from {self.input_spec} has an invalid shape"
                )
        if not len(self.image_in_keys):
            raise ValueError("CNN found no image inputs, maybe use an MLP?")
        if self.input_has_agent_dim and input_shape_image[-3] != self.n_agents:
            raise ValueError(
                "If the CNN input has the agent dimension,"
                " the forth to last spec dimension of image inputs should be the number of agents"
            )
        if (
            self.input_has_agent_dim
            and input_shape_tensor is not None
            and input_shape_tensor[-1] != self.n_agents
        ):
            raise ValueError(
                "If the CNN input has the agent dimension,"
                " the second to last spec dimension of tensor inputs should be the number of agents"
            )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the CNN output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather images
        input = torch.cat(
            [tensordict.get(in_key) for in_key in self.image_in_keys], dim=-1
        ).to(torch.float)
        # BenchMARL images are X,Y,C -> we convert them to C, X, Y for processing in TorchRL models
        input = input.transpose(-3, -1).transpose(-2, -1)

        # Gather tensor inputs
        if len(self.tensor_in_keys):
            tensor_inputs = torch.cat(
                [tensordict.get(in_key) for in_key in self.tensor_in_keys], dim=-1
            )
            if self.input_has_agent_dim and not self.output_has_agent_dim:
                tensor_inputs = tensor_inputs.reshape((*tensor_inputs.shape[:-2], -1))
            elif not self.input_has_agent_dim and self.output_has_agent_dim:
                tensor_inputs = tensor_inputs.unsqueeze(-2).expand(
                    (*tensor_inputs.shape[:-1], self.n_agents, tensor_inputs.shape[-1])
                )

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

        if len(self.tensor_in_keys):
            cnn_out = torch.cat([cnn_out, tensor_inputs], dim=-1)

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
    cnn_kernel_sizes: Union[Sequence[int], int] = MISSING
    cnn_strides: Union[Sequence[int], int] = MISSING
    cnn_paddings: Union[Sequence[int], int] = MISSING
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
