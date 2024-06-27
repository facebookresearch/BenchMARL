#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import importlib
from dataclasses import dataclass, MISSING
from math import prod
from typing import Optional, Sequence, Type

import torch
from tensordict import TensorDictBase
from torch import nn, Tensor

from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig

_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None
if _has_torch_geometric:
    import torch_geometric
    from torch_geometric.nn import MessagePassing
    from torch_geometric.transforms import BaseTransform

    class _RelVel(BaseTransform):
        """Transform that reads graph.vel and writes node1.vel - node2.vel in the edge attributes"""

        def __init__(self):
            pass

        def __call__(self, data):
            (row, col), vel, pseudo = data.edge_index, data.vel, data.edge_attr

            cart = vel[row] - vel[col]
            cart = cart.view(-1, 1) if cart.dim() == 1 else cart

            if pseudo is not None:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
            else:
                data.edge_attr = cart
            return data


TOPOLOGY_TYPES = {"full", "empty", "from_pos"}


class RmGnn(Model):
    """A GNN model.

    GNN models can be used as "decentralized" actors or critics.

    Args:
        topology (str): Topology of the graph adjacency matrix. Options: "full", "empty".
        self_loops (bool): Whether the resulting adjacency matrix will have self loops.
        gnn_class (Type[torch_geometric.nn.MessagePassing]): the gnn convolution class to use
        gnn_kwargs (dict, optional): the dict of arguments to pass to the gnn conv class

    Layer equation
        $$
        \mathbf{z}_i = \rho \left (\mathbf{x}_i \parallel   \bigoplus_{j \in \mathcal{N}_i} \phi \left ( \mathbf{x}_j \parallel  \mathbf{e}_{ij}  \right ) \right)
        $$

    """

    def __init__(
        self,
        topology: str,
        self_loops: bool,
        num_cells,
        activation_class,
        edge_radius: Optional[float],
        **kwargs,
    ):

        self.topology = topology
        self.self_loops = self_loops
        self.edge_radius = edge_radius

        super().__init__(**kwargs)

        self.hidden_size = 128
        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        self.gnns = nn.ModuleList(
            [
                MatPosConv(
                    in_dim=self.input_features - 2,  # input - pos,
                    out_dim=self.hidden_size,
                    edge_features=5,  # rel vel, rel pos and distance
                    aggr="sum",
                    activation_fn=activation_class,
                ).to(self.device)
                for _ in range(self.n_agents if not self.share_params else 1)
            ]
        )

        self.mlp_local_and_comms = MultiAgentMLP(
            n_agent_inputs=self.hidden_size + self.input_features - 2,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            num_cells=num_cells,
            activation_class=activation_class,
        )

        self.edge_index = _get_edge_index(
            topology=self.topology,
            self_loops=self.self_loops,
            device=self.device,
            n_agents=self.n_agents,
        )

    def _perform_checks(self):
        super()._perform_checks()

        if self.topology not in TOPOLOGY_TYPES:
            raise ValueError(
                f"Got topology: {self.topology} but only available options are {TOPOLOGY_TYPES}"
            )

        if not self.input_has_agent_dim:
            raise ValueError(
                "The GNN module is not compatible with input that does not have the agent dimension,"
                "such as the global state in centralised critics. Please choose another critic model"
                "if your algorithm has a centralized critic and the task has a global state."
            )

        input_shape = None
        for input_key, input_spec in self.input_spec.items(True, True):
            if len(input_spec.shape) == 2:
                if input_shape is None:
                    input_shape = input_spec.shape[:-1]
                else:
                    if input_spec.shape[:-1] != input_shape:
                        raise ValueError(
                            f"GNN inputs should all have the same shape up to the last dimension, got {self.input_spec}"
                        )
            else:
                raise ValueError(
                    f"GNN input value {input_key} from {self.input_spec} has an invalid shape"
                )

        if input_shape[-1] != self.n_agents:
            raise ValueError(
                f"The second to last input spec dimension should be the number of agents, got {self.input_spec}"
            )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the GNN output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        input = tensordict.get(self.in_key)

        batch_size = input.shape[:-2]

        pos = input[..., :2]
        vel = input[..., 2:4]
        input = input[..., 2:]  # exclude pos

        graph = _batch_from_dense_to_ptg(
            x=input,
            edge_index=self.edge_index,
            pos=pos,
            vel=vel,
            self_loops=self.self_loops,
            edge_radius=self.edge_radius,
        )

        if not self.share_params:
            gnn_out = torch.stack(
                [
                    gnn(graph.x, graph.edge_index, graph.edge_attr).view(
                        *batch_size,
                        self.n_agents,
                        self.hidden_size,
                    )[..., i, :]
                    for i, gnn in enumerate(self.gnns)
                ],
                dim=-2,
            )
        else:
            gnn_out = self.gnns[0](
                graph.x,
                graph.edge_index,
                graph.edge_attr,
            ).view(*batch_size, self.n_agents, self.hidden_size)

        res = self.mlp_local_and_comms(torch.cat([input, gnn_out], dim=-1))

        tensordict.set(self.out_key, res)
        return tensordict


class MatPosConv(MessagePassing):
    propagate_type = {"x": Tensor, "edge_attr": Tensor}

    def __init__(self, in_dim, out_dim, edge_features, **cfg):
        super().__init__(aggr=cfg["aggr"])

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_features = edge_features
        self.activation_fn = cfg["activation_fn"]

        self.message_encoder = nn.Sequential(
            torch.nn.Linear(self.in_dim + self.edge_features, self.out_dim),
            self.activation_fn(),
            torch.nn.Linear(self.out_dim, self.out_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        msg = self.message_encoder(torch.cat([x_j, edge_attr], dim=-1))
        return msg

    def update(self, inputs: Tensor, x) -> Tensor:
        return inputs


def _get_edge_index(topology: str, self_loops: bool, n_agents: int, device: str):
    if topology == "full":
        adjacency = torch.ones(n_agents, n_agents, device=device, dtype=torch.long)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adjacency)
        if not self_loops:
            edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
    elif topology == "empty":
        if self_loops:
            edge_index = (
                torch.arange(n_agents, device=device, dtype=torch.long)
                .unsqueeze(0)
                .repeat(2, 1)
            )
        else:
            edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
    elif topology == "from_pos":
        edge_index = None
    else:
        raise ValueError(f"Topology {topology} not supported")

    return edge_index


class RelVel(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        (row, col), vel, pseudo = data.edge_index, data.vel, data.edge_attr

        cart = vel[row] - vel[col]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if pseudo is not None:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data


def _batch_from_dense_to_ptg(
    x: Tensor,
    edge_index: Optional[Tensor],
    self_loops: bool,
    pos: Tensor = None,
    vel: Tensor = None,
    edge_radius: Optional[float] = None,
) -> torch_geometric.data.Batch:
    batch_size = prod(x.shape[:-2])
    n_agents = x.shape[-2]
    x = x.view(-1, x.shape[-1])
    if pos is not None:
        pos = pos.view(-1, pos.shape[-1])
    if vel is not None:
        vel = vel.view(-1, vel.shape[-1])

    b = torch.arange(batch_size, device=x.device)

    graphs = torch_geometric.data.Batch()
    graphs.ptr = torch.arange(0, (batch_size + 1) * n_agents, n_agents)
    graphs.batch = torch.repeat_interleave(b, n_agents)
    graphs.x = x
    graphs.pos = pos
    graphs.vel = vel
    graphs.edge_attr = None

    if edge_index is not None:
        n_edges = edge_index.shape[1]
        # Tensor of shape [batch_size * n_edges]
        # in which edges corresponding to the same graph have the same index.
        batch = torch.repeat_interleave(b, n_edges)
        # Edge index for the batched graphs of shape [2, n_edges * batch_size]
        # we sum to each batch an offset of batch_num * n_agents to make sure that
        # the adjacency matrices remain independent
        batch_edge_index = edge_index.repeat(1, batch_size) + batch * n_agents
        graphs.edge_index = batch_edge_index
    else:
        if pos is None:
            raise RuntimeError("from_pos topology needs positions as input")
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(
            graphs.pos, batch=graphs.batch, r=edge_radius, loop=self_loops
        )

    graphs = graphs.to(x.device)
    if pos is not None:
        graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs)
        graphs = torch_geometric.transforms.Distance(norm=False)(graphs)
    if vel is not None:
        graphs = _RelVel()(graphs)

    return graphs


@dataclass
class RmGnnConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Gnn`."""

    topology: str = MISSING
    self_loops: bool = MISSING

    num_cells: Sequence[int] = MISSING
    activation_class: Type[nn.Module] = MISSING
    edge_radius: Optional[float] = None

    @staticmethod
    def associated_class():
        return RmGnn
