#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

import torch

from tensordict import TensorDictBase
from torch import nn
from torchrl.modules.models import DdpgCnnActor as CnnActor

from benchmarl.models.common import Model, ModelConfig
from benchmarl.utils import DEVICE_TYPING


class MultiAgentCnnActor(nn.Module):
    """Multi-agent CNN.

    In MARL settings, agents may or may not share the same policy for their actions: we say that the parameters can be shared or not. Similarly, a network may take the entire observation space (across agents) or on a per-agent basis to compute its output, which we refer to as "centralized" and "non-centralized", respectively.

    It expects inputs with shape ``(*B, n_agents, channels, x, y)``.
    Args:
        n_agents (int): number of agents.
        centralised (bool): If ``True``, each agent will use the inputs of all agents to compute its output, resulting in input of shape ``(*B, n_agents * channels, x, y)``. Otherwise, each agent will only use its data as input.
        share_params (bool): If ``True``, the same :class:`~torchrl.modules.ConvNet` will be used to make the forward pass
            for all agents (homogeneous policies). Otherwise, each agent will use a different :class:`~torchrl.modules.ConvNet` to process
            its input (heterogeneous policies).
        device (str or torch.device, optional): device to create the module on.
        num_cells (int or Sequence[int], optional): number of cells of every layer in between the input and output. If
            an integer is provided, every layer will have the same number of cells. If an iterable is provided,
            the linear layers ``out_features`` will match the content of ``num_cells``.
        kernel_sizes (int, Sequence[Union[int, Sequence[int]]]): Kernel size(s) of the convolutional network.
            Defaults to ``5``.
        strides (int or Sequence[int]): Stride(s) of the convolutional network. If iterable, the length must match the
            depth, defined by the num_cells or depth arguments.
            Defaults to ``2``.
        activation_class (Type[nn.Module]): activation class to be used.
            Default to :class:`torch.nn.ELU`.
        **kwargs: for :class:`~torchrl.modules.models.ConvNet` can be passed to customize the ConvNet.

    """

    def __init__(
        self,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        action_dim: int,
        device: Optional[DEVICE_TYPING] = None,  # type: ignore
        conv_net_kwargs: Optional[dict] = None,
        mlp_net_kwargs: Optional[dict] = None,
        use_avg_pooling: bool = False,
        n_agent_inputs: int = None,
        **kwargs,
    ):
        super().__init__()

        if n_agent_inputs is not None and not centralised:
            raise ValueError("n_agents input can only be set for decentralised")

        if n_agent_inputs is None:
            n_agent_inputs = n_agents

        self.n_agent_inputs = n_agent_inputs
        self.n_agents = n_agents
        self.centralised = centralised
        self.share_params = share_params

        self.agent_networks = nn.ModuleList(
            [
                CnnActor(
                    action_dim=action_dim,
                    conv_net_kwargs=conv_net_kwargs,
                    mlp_net_kwargs=mlp_net_kwargs,
                    use_avg_pooling=use_avg_pooling,
                    device=device,
                )
                for _ in range(self.n_agents if not self.share_params else 1)
            ]
        )

    def forward(self, inputs: torch.Tensor):
        if len(inputs.shape) < 4:
            raise ValueError(
                """Multi-agent network expects (*batch_size, num_agents, channels, x, y)"""
            )
        if inputs.shape[-4] != self.n_agent_inputs:
            raise ValueError(
                f"""Multi-agent network expects {self.n_agent_inputs} but got {inputs.shape[-4]}"""
            )
        # If the model is centralized, agents have full observability
        if self.centralised:
            shape = (
                *inputs.shape[:-4],
                self.n_agent_inputs
                * inputs.shape[-3],  # multiply channels by num_agents
                inputs.shape[-2],  # x
                inputs.shape[-1],  # y
            )
            inputs = torch.reshape(inputs, shape)

        # If the parameters are not shared, each agent has its own network
        if not self.share_params:
            if self.centralised:
                action = torch.stack(
                    [net(inputs)[0] for net in self.agent_networks], dim=-2
                )
            else:
                action = torch.stack(
                    [
                        net(inp)[0]
                        for i, (net, inp) in enumerate(
                            zip(self.agent_networks, inputs.unbind(-4))
                        )
                    ],
                    dim=-2,
                )
        else:
            action, hidden = self.agent_networks[0](inputs)
            if self.centralised:
                # If the parameters are shared, and it is centralised all agents will have the same action.
                # We expand it to maintain the agent dimension, but values will be the same for all agents
                n_agent_actions = action.shape[-1]
                action = action.view(*action.shape[:-1], n_agent_actions)
                action = action.unsqueeze(-2)
                action = action.expand(
                    *action.shape[:-2], self.n_agents, n_agent_actions
                )
        return action


class Cnn(Model):
    """CNN Actor model.

    Args:
        num_cells (int or Sequence[int], optional): number of cells of every layer in between the input and output. If
            an integer is provided, every layer will have the same number of cells. If an iterable is provided,
            the linear layers out_features will match the content of num_cells.
        layer_class (Type[nn.Module]): class to be used for the linear layers;
        activation_class (Type[nn.Module]): activation class to be used.
        activation_kwargs (dict, optional): kwargs to be used with the activation class;
        norm_class (Type, optional): normalization class, if any.
        norm_kwargs (dict, optional): kwargs to be used with the normalization layers;

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
        )

        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        mlp_net_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("mlp_")}
        cnn_net_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("cnn_")}

        self.cnn = MultiAgentCnnActor(
            n_agent_inputs=self.input_features if self.centralised else None,
            action_dim=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            conv_net_kwargs=cnn_net_kwargs,
            mlp_net_kwargs=mlp_net_kwargs,
            **kwargs,
        )

        # initialize lazy parameters with input observation sizes
        self.cnn(
            torch.randn(
                (
                    1,
                    self.n_agents,
                    kwargs["in_features"],
                    kwargs["height"],
                    kwargs["width"],
                )
            ).to(self.device)
        )

    def _perform_checks(self):
        super()._perform_checks()
        # TODO add cnn checks

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        input = tensordict.get(self.in_key)

        # Has multi-agent input dimension
        res = self.cnn.forward(input)
        if not self.output_has_agent_dim:
            # If we are here the module is centralised and parameter shared.
            # Thus the multi-agent dimension has been expanded,
            # We remove it without loss of data
            res = res[..., 0, :]

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class CnnConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Cnn`."""

    height: int = MISSING
    width: int = MISSING
    in_features: int = MISSING

    cnn_num_cells: Sequence[int] = MISSING
    cnn_kernel_sizes: Sequence[int] = MISSING
    cnn_strides: Sequence[int] = MISSING
    cnn_paddings: Sequence[int] = MISSING

    cnn_activation_class: Type[nn.Module] = MISSING
    cnn_aggregator_class: Type[nn.Module] = MISSING
    cnn_aggregator_kwargs: Optional[dict] = MISSING
    cnn_squeeze_output: bool = MISSING

    mlp_depth: int = MISSING
    mlp_num_cells: Sequence[int] = MISSING
    mlp_layer_class: Type[nn.Module] = MISSING

    mlp_activation_class: Type[nn.Module] = MISSING
    mlp_activation_kwargs: Optional[dict] = None

    mlp_norm_class: Type[nn.Module] = None
    mlp_norm_kwargs: Optional[dict] = None

    mlp_bias_last_layer: bool = True

    @staticmethod
    def associated_class():
        return Cnn
