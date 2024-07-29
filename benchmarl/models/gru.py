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
from typing import Callable, Optional, Sequence, Type

import torch
from tensordict import TensorDictBase
from tensordict.utils import expand_as_right
from torch import nn
from torchrl.data.tensor_specs import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import Compose, EnvBase, InitTracker, TensorDictPrimer, TransformedEnv
from torchrl.modules import GRUCell, MLP, MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig


class MultiAgentGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_agents, device):
        super().__init__()
        self.input_size = input_size
        self.n_agents = n_agents
        self.hidden_size = hidden_size
        self.device = device

        self.gru = GRUCell(input_size, hidden_size, device=self.device)

        self.vmap_rnn = self.get_for_loop(self.gru)
        # self.vmap_rnn_compiled = torch.compile(
        #     self.vmap_rnn, mode="reduce-overhead", fullgraph=True
        # )

    def forward(
        self,
        input,
        is_init,
        h_0=None,
    ):
        assert is_init is not None, "We need to pass is_init"
        training = h_0 is None
        if (
            not training
        ):  # In collection we emulate the sequence dimension and we have the hidden state
            input = input.unsqueeze(1)

        # Check input
        batch = input.shape[0]
        seq = input.shape[1]
        assert input.shape == (batch, seq, self.n_agents, self.input_size)

        if h_0 is not None:  # Collection
            assert h_0.shape == (
                batch,
                self.n_agents,
                self.hidden_size,
            )
            if is_init is not None:  # Set hidden to 0 when is_init
                h_0 = torch.where(expand_as_right(is_init, h_0), 0, h_0)

        if not training:  # If in collection emulate the sequence dimension
            is_init = is_init.unsqueeze(1)
        assert is_init.shape == (batch, seq, 1)
        is_init = is_init.unsqueeze(-2).expand(batch, seq, self.n_agents, 1)

        if h_0 is None:
            h_0 = torch.zeros(
                batch,
                self.n_agents,
                self.hidden_size,
                device=self.device,
                dtype=torch.float,
            )
        output = self.vmap_rnn(input, is_init, h_0)
        h_n = output[..., -1, :, :]

        if not training:
            output = output.squeeze(1)
        return output, h_n

    # @torch.compile(mode="reduce-overhead", fullgraph=True)

    @staticmethod
    def get_for_loop(rnn):
        def for_loop(input, is_init, h, time_dim=-3):
            hs = []
            for in_t, init_t in zip(input.unbind(time_dim), is_init.unbind(time_dim)):
                h = torch.where(init_t, 0, h)
                h = rnn(in_t, h)
                hs.append(h)
            output = torch.stack(hs, time_dim)
            return output

        return torch.vmap(for_loop)


class Gru(Model):
    def __init__(
        self,
        hidden_size: int,
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

        self.hidden_size = hidden_size

        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        )
        self.output_features = self.output_leaf_spec.shape[-1]

        if self.input_has_agent_dim:
            self.gru = MultiAgentGRU(
                self.input_features, self.hidden_size, self.n_agents, self.device
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
        input = torch.cat([tensordict.get(in_key) for in_key in self.in_keys], dim=-1)
        h_0 = tensordict.get((self.agent_group, "_hidden_gru"), None)
        is_init = tensordict.get("is_init")

        # Has multi-agent input dimension
        if self.input_has_agent_dim and self.share_params and not self.centralised:
            output, h_n = self.gru(input, is_init, h_0)
        else:
            pass

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
        if h_0 is not None:
            tensordict.set(("next", self.agent_group, "_hidden_gru"), h_n)
        return tensordict


@dataclass
class GruConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Gru`."""

    hidden_size: int = MISSING

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

    def process_env_fun(
        self,
        env_fun: Callable[[], EnvBase],
        task,
        model_index: int = 0,
    ) -> Callable[[], EnvBase]:
        """
        This function can be used to wrap env_fun
        Args:
            env_fun (callable): a function that takes no args and creates an enviornment

        Returns: a function that takes no args and creates an enviornment

        """

        def model_fun():
            env = env_fun()
            env = TransformedEnv(
                env,
                Compose(
                    InitTracker(init_key="is_init"),
                    TensorDictPrimer(
                        {
                            group: CompositeSpec(
                                {
                                    "_hidden_gru": UnboundedContinuousTensorSpec(
                                        shape=(len(agents), self.hidden_size)
                                    )
                                },
                                shape=(len(agents),),
                            )
                            for group, agents in task.group_map(env).items()
                        }
                    ),
                ),
            )
            return env

        return model_fun
