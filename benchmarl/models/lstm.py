#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import expand_as_right, unravel_key_list
from torch import nn
from torchrl.data.tensor_specs import Composite, Unbounded

from torchrl.modules import LSTMCell, MLP, MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from benchmarl.utils import DEVICE_TYPING


class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        device: DEVICE_TYPING,
        n_layers: int,
        dropout: float,
        bias: bool,
        time_dim: int = -2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.time_dim = time_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bias = bias

        self.lstms = torch.nn.ModuleList(
            [
                LSTMCell(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    device=self.device,
                    bias=self.bias,
                )
                for i in range(self.n_layers)
            ]
        )

    def forward(self, input, is_init, h, c):
        hs = []

        h = list(h.unbind(dim=-2))
        c = list(c.unbind(dim=-2))

        for in_t, init_t in zip(
            input.unbind(self.time_dim), is_init.unbind(self.time_dim)
        ):
            for layer in range(self.n_layers):
                h[layer] = torch.where(init_t, 0, h[layer])
                c[layer] = torch.where(init_t, 0, c[layer])

                h[layer], c[layer] = self.lstms[layer](in_t, (h[layer], c[layer]))

                if layer < self.n_layers - 1 and self.dropout:
                    in_t = F.dropout(h[layer], p=self.dropout, training=self.training)
                else:
                    in_t = h[layer]

            hs.append(in_t)
        h_n = torch.stack(h, dim=-2)
        c_n = torch.stack(c, dim=-2)
        output = torch.stack(hs, self.time_dim)

        return output, h_n, c_n


def get_net(input_size, hidden_size, n_layers, bias, device, dropout, compile):
    lstm = LSTM(
        input_size,
        hidden_size,
        n_layers=n_layers,
        bias=bias,
        device=device,
        dropout=dropout,
    )
    if compile:
        lstm = torch.compile(lstm, mode="reduce-overhead")
    return lstm


class MultiAgentLSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_agents: int,
        device: DEVICE_TYPING,
        centralised: bool,
        share_params: bool,
        n_layers: int,
        dropout: float,
        bias: bool,
        compile: bool,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_agents = n_agents
        self.hidden_size = hidden_size
        self.device = device
        self.centralised = centralised
        self.share_params = share_params
        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.compile = compile

        if self.centralised:
            input_size = input_size * self.n_agents

        agent_networks = [
            get_net(
                input_size=input_size,
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                bias=self.bias,
                device=self.device,
                dropout=self.dropout,
                compile=self.compile,
            )
            for _ in range(self.n_agents if not self.share_params else 1)
        ]
        self._make_params(agent_networks)

        with torch.device("meta"):
            self._empty_lstm = get_net(
                input_size=input_size,
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                bias=self.bias,
                device="meta",
                dropout=self.dropout,
                compile=self.compile,
            )
            # Remove all parameters
            TensorDict.from_module(self._empty_lstm).data.to("meta").to_module(
                self._empty_lstm
            )

    def forward(
        self,
        input,
        is_init,
        h_0=None,
        c_0=None,
    ):
        # Input and output always have the multiagent dimension
        # Hidden states always have it apart from when it is centralized and share params
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
            c_0 = c_0.unsqueeze(0)
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
            c_0 = torch.where(expand_as_right(is_init, c_0), 0, c_0)

        if not training:  # If in collection emulate the sequence dimension
            is_init = is_init.unsqueeze(1)
        assert is_init.shape == (batch, seq, 1)
        is_init = is_init.unsqueeze(-2).expand(batch, seq, self.n_agents, 1)

        if h_0 is None:
            if self.centralised and self.share_params:
                shape = (
                    batch,
                    self.n_layers,
                    self.hidden_size,
                )
            else:
                shape = (
                    batch,
                    self.n_agents,
                    self.n_layers,
                    self.hidden_size,
                )
            h_0 = torch.zeros(
                shape,
                device=self.device,
                dtype=torch.float,
            )
            c_0 = h_0.clone()
        if self.centralised:
            input = input.view(batch, seq, self.n_agents * self.input_size)
            is_init = is_init[..., 0, :]

        output, h_n, c_n = self.run_net(input, is_init, h_0, c_0)

        if self.centralised and self.share_params:
            output = output.unsqueeze(-2).expand(
                batch, seq, self.n_agents, self.hidden_size
            )

        if not training:
            output = output.squeeze(1)
        if missing_batch:
            output = output.squeeze(0)
            h_n = h_n.squeeze(0)
            c_n = c_n.squeeze(0)
        return output, h_n, c_n

    def run_net(self, input, is_init, h_0, c_0):
        if not self.share_params:
            if self.centralised:
                output, h_n, c_n = self.vmap_func_module(
                    self._empty_lstm,
                    (0, None, None, -3, -3),
                    (-2, -3, -3),
                )(self.params, input, is_init, h_0, c_0)
            else:
                output, h_n, c_n = self.vmap_func_module(
                    self._empty_lstm,
                    (0, -2, -2, -3, -3),
                    (-2, -3, -3),
                )(self.params, input, is_init, h_0, c_0)
        else:
            with self.params.to_module(self._empty_lstm):
                if self.centralised:
                    output, h_n, c_n = self._empty_lstm(input, is_init, h_0, c_0)
                else:
                    output, h_n, c_n = torch.vmap(
                        self._empty_lstm,
                        in_dims=(-2, -2, -3, -3),
                        out_dims=(-2, -3, -3),
                    )(input, is_init, h_0, c_0)

        return output, h_n, c_n

    def vmap_func_module(self, module, *args, **kwargs):
        def exec_module(params, *input):
            with params.to_module(module):
                return module(*input)

        return torch.vmap(exec_module, *args, **kwargs)

    def _make_params(self, agent_networks):
        if self.share_params:
            self.params = TensorDict.from_module(agent_networks[0], as_module=True)
        else:
            self.params = TensorDict.from_modules(*agent_networks, as_module=True)


class Lstm(Model):
    r"""A multi-layer Long Short-Term Memory (LSTM) RNN like the one from
    `torch <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>`__ .

    The BenchMARL LSTM accepts multiple inputs of type array: Tensors of shape ``(*batch,F)``

    Where `F` is the number of features. These arrays will be concatenated along the F dimensions,
    which will be processed to features of `hidden_size` by the LSTM.

    Args:
        hidden_size (int): The number of features in the hidden state.
        num_layers (int): Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two lstms together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias (bool): If ``False``, then the LSTM layers do not use bias.
            Default: ``True``
        dropout (float): If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        compile (bool): If ``True``, compiles underlying LSTM model. Default: ``False``

    """

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        bias: bool,
        dropout: float,
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

        self.hidden_state_name_h = (
            self.agent_group,
            f"_hidden_lstm_h_{self.model_index}",
        )
        self.hidden_state_name_c = (
            self.agent_group,
            f"_hidden_lstm_c_{self.model_index}",
        )

        self.rnn_keys = unravel_key_list(
            ["is_init", self.hidden_state_name_c, self.hidden_state_name_h]
        )
        self.in_keys += self.rnn_keys

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.compile = compile

        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        )
        self.output_features = self.output_leaf_spec.shape[-1]

        if self.input_has_agent_dim:
            self.lstm = MultiAgentLSTM(
                self.input_features,
                self.hidden_size,
                self.n_agents,
                self.device,
                bias=self.bias,
                n_layers=self.n_layers,
                centralised=self.centralised,
                share_params=self.share_params,
                dropout=self.dropout,
                compile=self.compile,
            )
        else:
            self.lstm = nn.ModuleList(
                [
                    get_net(
                        input_size=self.input_features,
                        hidden_size=self.hidden_size,
                        n_layers=self.n_layers,
                        bias=self.bias,
                        device=self.device,
                        dropout=self.dropout,
                        compile=self.compile,
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
                            f"LSTM inputs should all have the same shape up to the last dimension, got {self.input_spec}"
                        )
            else:
                raise ValueError(
                    f"LSTM input value {input_key} from {self.input_spec} has an invalid shape, maybe you need a CNN?"
                )
        if self.input_has_agent_dim:
            if input_shape[-1] != self.n_agents:
                raise ValueError(
                    "If the LSTM input has the agent dimension,"
                    f" the second to last spec dimension should be the number of agents, got {self.input_spec}"
                )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the LSTM output has the agent dimension,"
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
        h_0 = tensordict.get(self.hidden_state_name_h, None)
        c_0 = tensordict.get(self.hidden_state_name_c, None)
        is_init = tensordict.get("is_init")

        training = h_0 is None

        # Has multi-agent input dimension
        if self.input_has_agent_dim:
            output, h_n, c_n = self.lstm(input, is_init, h_0, c_0)
            if not self.output_has_agent_dim:
                output = output[..., 0, :]
        else:  # Is a global input, this is a critic
            # Check input
            batch = input.shape[0]
            seq = input.shape[1]
            assert input.shape == (batch, seq, self.input_features)
            assert is_init.shape == (batch, seq, 1)

            h_0 = torch.zeros(
                (batch, self.n_layers, self.hidden_size),
                device=self.device,
                dtype=torch.float,
            )
            c_0 = h_0.clone()
            if self.share_params:
                output, _, _ = self.lstm[0](input, is_init, h_0, c_0)
            else:
                outputs = []
                for net in self.lstm:
                    output, _, _ = net(input, is_init, h_0, c_0)
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
            tensordict.set(("next", *self.hidden_state_name_h), h_n)
            tensordict.set(("next", *self.hidden_state_name_c), c_n)
        return tensordict


@dataclass
class LstmConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.LSTM`."""

    hidden_size: int = MISSING
    n_layers: int = MISSING
    bias: bool = MISSING
    dropout: float = MISSING
    compile: bool = MISSING

    mlp_num_cells: Sequence[int] = MISSING
    mlp_layer_class: Type[nn.Module] = MISSING
    mlp_activation_class: Type[nn.Module] = MISSING

    mlp_activation_kwargs: Optional[dict] = None
    mlp_norm_class: Type[nn.Module] = None
    mlp_norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return Lstm

    @property
    def is_rnn(self) -> bool:
        return True

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        spec = Composite(
            {
                f"_hidden_lstm_c_{model_index}": Unbounded(
                    shape=(self.n_layers, self.hidden_size)
                ),
                f"_hidden_lstm_h_{model_index}": Unbounded(
                    shape=(self.n_layers, self.hidden_size)
                ),
            }
        )
        return spec
