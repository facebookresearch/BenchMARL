#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

import torch
from tensordict import TensorDictBase
from tensordict.utils import expand_as_right, unravel_key_list
from torch import nn
from torchrl.data.tensor_specs import CompositeSpec, UnboundedContinuousTensorSpec

from torchrl.modules import GRUCell, MLP, MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from benchmarl.utils import DEVICE_TYPING


class GRU(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        device: DEVICE_TYPING,
        time_dim: int = -2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.time_dim = time_dim

        self.gru = GRUCell(input_size, hidden_size, device=self.device)

    def forward(
        self,
        input,
        is_init,
        h,
    ):
        hs = []
        for in_t, init_t in zip(
            input.unbind(self.time_dim), is_init.unbind(self.time_dim)
        ):
            h = torch.where(init_t, 0, h)
            h = self.gru(in_t, h)
            hs.append(h)
        h_n = h
        output = torch.stack(hs, self.time_dim)

        return output, h_n


class MultiAgentGRU(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_agents: int,
        device: DEVICE_TYPING,
        compile: bool,
        centralised: bool,
        share_params: bool,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_agents = n_agents
        self.hidden_size = hidden_size
        self.device = device
        self.compile = compile
        self.centralised = centralised

        if not share_params:
            raise NotImplementedError

        if self.centralised:
            input_size = input_size * self.n_agents

        self.base_gru = GRU(
            input_size,
            hidden_size,
            device=self.device,
        )

        if self.compile:
            self.base_gru = torch.compile(
                self.base_gru, mode="reduce-overhead", fullgraph=True
            )
        if not self.centralised:
            self.gru = torch.vmap(self.base_gru, in_dims=-2, out_dims=-2)
        else:
            self.gru = self.base_gru

    def forward(
        self,
        input,
        is_init,
        h_0=None,
    ):
        # Input and output always have the multiagent dimension
        # Hidden state only has it when not centralised
        # is_init never has it

        assert is_init is not None, "We need to pass is_init"
        training = h_0 is None

        missing_batch = False
        if (
            not training and len(input.shape) < 3
        ):  # In evaluation the batch might be missing
            missing_batch = True
            input = input.unsqueeze(0)
            h_0 = h_0.unsqueeze(0)
            is_init = is_init.unsqueeze(0)

        if (
            not training
        ):  # In collection we emulate the sequence dimension and we have the hidden state
            input = input.unsqueeze(1)

        # Check input
        batch = input.shape[0]
        seq = input.shape[1]
        assert input.shape == (batch, seq, self.n_agents, self.input_size)

        if h_0 is not None:  # Collection
            # Set hidden to 0 when is_init
            h_0 = torch.where(expand_as_right(is_init, h_0), 0, h_0)

        if not training:  # If in collection emulate the sequence dimension
            is_init = is_init.unsqueeze(1)
        assert is_init.shape == (batch, seq, 1)
        is_init = is_init.unsqueeze(-2).expand(batch, seq, self.n_agents, 1)

        if h_0 is None:
            if self.centralised:
                shape = (
                    batch,
                    self.hidden_size,
                )
            else:
                shape = (
                    batch,
                    self.n_agents,
                    self.hidden_size,
                )
            h_0 = torch.zeros(
                shape,
                device=self.device,
                dtype=torch.float,
            )
        if self.centralised:
            input = input.view(batch, seq, self.n_agents * self.input_size)
            is_init = is_init[..., 0, :]

        output, h_n = self.gru(input, is_init, h_0)

        if self.centralised:
            output = output.unsqueeze(-2).expand(
                batch, seq, self.n_agents, self.hidden_size
            )

        if not training:
            output = output.squeeze(1)
        if missing_batch:
            output = output.squeeze(0)
            h_n = h_n.squeeze(0)
        return output, h_n


class Gru(Model):
    def __init__(
        self,
        hidden_size: int,
        compile: bool,
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

        self.hidden_state_name = (self.agent_group, f"_hidden_gru_{self.model_index}")
        self.rnn_keys = unravel_key_list(["is_init", self.hidden_state_name])
        self.in_keys += self.rnn_keys

        self.hidden_size = hidden_size
        self.compile = compile

        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        )
        self.output_features = self.output_leaf_spec.shape[-1]

        if self.input_has_agent_dim:
            self.gru = MultiAgentGRU(
                self.input_features,
                self.hidden_size,
                self.n_agents,
                self.device,
                centralised=self.centralised,
                share_params=self.share_params,
                compile=self.compile,
            )
        else:
            self.gru = nn.ModuleList(
                [
                    GRU(
                        self.input_features,
                        self.hidden_size,
                        device=self.device,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

        mlp_net_kwargs = {
            "_".join(k.split("_")[1:]): v
            for k, v in kwargs.items()
            if k.startswith("mlp_")
        }
        if self.output_has_agent_dim:
            self.mlp = MultiAgentMLP(
                n_agent_inputs=self.hidden_size,
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
                        in_features=self.hidden_size,
                        out_features=self.output_features,
                        device=self.device,
                        **mlp_net_kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

    def _perform_checks(self):
        super()._perform_checks()

        input_shape = None
        for input_key, input_spec in self.input_spec.items(True, True):
            if (self.input_has_agent_dim and len(input_spec.shape) == 2) or (
                not self.input_has_agent_dim and len(input_spec.shape) == 1
            ):
                if input_shape is None:
                    input_shape = input_spec.shape[:-1]
                else:
                    if input_spec.shape[:-1] != input_shape:
                        raise ValueError(
                            f"GRU inputs should all have the same shape up to the last dimension, got {self.input_spec}"
                        )
            else:
                raise ValueError(
                    f"GRU input value {input_key} from {self.input_spec} has an invalid shape, maybe you need a CNN?"
                )
        if self.input_has_agent_dim:
            if input_shape[-1] != self.n_agents:
                raise ValueError(
                    "If the GRU input has the agent dimension,"
                    f" the second to last spec dimension should be the number of agents, got {self.input_spec}"
                )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the GRU output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        input = torch.cat(
            [
                tensordict.get(in_key)
                for in_key in self.in_keys
                if in_key not in self.rnn_keys
            ],
            dim=-1,
        )
        h_0 = tensordict.get(self.hidden_state_name, None)
        is_init = tensordict.get("is_init")
        training = h_0 is None

        # Has multi-agent input dimension
        if self.input_has_agent_dim:
            output, h_n = self.gru(input, is_init, h_0)
            if not self.output_has_agent_dim:
                output = output[..., 0, :]
        else:  # Is a global input, this is a critic
            # Check input
            assert training and self.is_critic
            batch = input.shape[0]
            seq = input.shape[1]
            assert input.shape == (batch, seq, self.input_features)
            assert is_init.shape == (batch, seq, 1)

            h_0 = torch.zeros(
                (batch, self.hidden_size),
                device=self.device,
                dtype=torch.float,
            )
            if self.share_params:
                output, _ = self.gru[0](input, is_init, h_0)
            else:
                outputs = []
                for net in self.gru:
                    output, _ = net(input, is_init, h_0)
                    outputs.append(output)
                output = torch.stack(outputs, dim=-2)

        # Mlp
        if self.output_has_agent_dim:
            output = self.mlp.forward(output)
        else:
            if not self.share_params:
                output = torch.stack(
                    [net(output) for net in self.mlp],
                    dim=-2,
                )
            else:
                output = self.mlp[0](output)

        tensordict.set(self.out_key, output)
        if not training:
            tensordict.set(("next", *self.hidden_state_name), h_n)
        return tensordict


@dataclass
class GruConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Gru`."""

    hidden_size: int = MISSING
    compile: bool = MISSING

    mlp_num_cells: Sequence[int] = MISSING
    mlp_layer_class: Type[nn.Module] = MISSING
    mlp_activation_class: Type[nn.Module] = MISSING

    mlp_activation_kwargs: Optional[dict] = None
    mlp_norm_class: Type[nn.Module] = None
    mlp_norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return Gru

    @property
    def is_rnn(self) -> bool:
        return True

    def get_model_state_spec(self, model_index: int = 0) -> CompositeSpec:
        name = f"_hidden_gru_{model_index}"
        spec = CompositeSpec(
            {name: UnboundedContinuousTensorSpec(shape=(self.hidden_size,))}
        )
        return spec
