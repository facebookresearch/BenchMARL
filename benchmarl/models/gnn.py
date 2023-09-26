from __future__ import annotations

import importlib
from dataclasses import dataclass
from math import prod
from typing import Optional

import torch
from tensordict import TensorDictBase
from torch import nn, Tensor

from benchmarl.models.common import Model, ModelConfig, parse_model_config
from benchmarl.utils import read_yaml_config

_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None
if _has_torch_geometric:
    import torch_geometric
    from torch_geometric.nn import GATv2Conv, GINEConv, GraphConv


class Gnn(Model):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        self.gnns = nn.ModuleList(
            [
                GnnKernel(
                    in_dim=self.input_features,
                    out_dim=self.output_features,
                )
                for _ in range(self.n_agents if not self.share_params else 1)
            ]
        )
        self.fully_connected_adjacency = torch.ones(
            self.n_agents, self.n_agents, device=self.device
        )

    def _perform_checks(self):
        super()._perform_checks()
        if not self.input_has_agent_dim:
            raise ValueError(
                "The GNN module is not compatible with input that does not have the agent dimension,"
                "such as the global state in centralised critics. Please choose another critic model"
                "if your algorithm has a centralized critic and the task has a global state."
            )

        if self.input_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError(
                "The second to last input spec dimension should be the number of agents"
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

        # For now fully connected
        adjacency = self.fully_connected_adjacency.to(input.device)

        edge_index, _ = torch_geometric.utils.dense_to_sparse(adjacency)
        edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)

        batch_size = input.shape[:-2]

        graph = batch_from_dense_to_ptg(x=input, edge_index=edge_index)

        if not self.share_params:
            res = torch.stack(
                [
                    gnn(graph.x, graph.edge_index).view(
                        *batch_size,
                        self.n_agents,
                        self.output_features,
                    )[:, i]
                    for i, gnn in enumerate(self.gnns)
                ],
                dim=-2,
            )

        else:
            res = self.gnns[0](
                graph.x,
                graph.edge_index,
            ).view(*batch_size, self.n_agents, self.output_features)

        tensordict.set(self.out_key, res)
        return tensordict


class GnnKernel(nn.Module):
    def __init__(self, in_dim, out_dim, **cfg):
        super().__init__()

        # gnn_types = {"GraphConv", "GATv2Conv", "GINEConv"}
        # aggr_types = {"add", "mean", "max"}

        self.aggr = "add"
        self.gnn_type = "GraphConv"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_fn = nn.Tanh

        if self.gnn_type == "GraphConv":
            self.gnn = GraphConv(
                self.in_dim,
                self.out_dim,
                aggr=self.aggr,
            )
        elif self.gnn_type == "GATv2Conv":
            # Default adds self loops
            self.gnn = GATv2Conv(
                self.in_dim,
                self.out_dim,
                edge_dim=self.edge_features,
                fill_value=0.0,
                share_weights=True,
                add_self_loops=True,
                aggr=self.aggr,
            )
        elif self.gnn_type == "GINEConv":
            self.gnn = GINEConv(
                nn=nn.Sequential(
                    torch.nn.Linear(self.in_dim, self.out_dim),
                    self.activation_fn(),
                ),
                edge_dim=self.edge_features,
                aggr=self.aggr,
            )

    def forward(self, x, edge_index):
        out = self.gnn(x, edge_index)
        return out


def batch_from_dense_to_ptg(
    x: Tensor,
    edge_index: Tensor,
) -> torch_geometric.data.Batch:

    batch_size = prod(x.shape[:-2])
    n_agents = x.shape[-2]
    x = x.view(-1, x.shape[-1])

    b = torch.arange(batch_size, device=x.device)

    graphs = torch_geometric.data.Batch()
    graphs.ptr = torch.arange(0, (batch_size + 1) * n_agents, n_agents)
    graphs.batch = torch.repeat_interleave(b, n_agents)
    graphs.x = x
    graphs.edge_attr = None

    n_edges = edge_index.shape[1]
    # Tensor of shape [batch_size * n_edges]
    # in which edges corresponding to the same graph have the same index.
    batch = torch.repeat_interleave(b, n_edges)
    # Edge index for the batched graphs of shape [2, n_edges * batch_size]
    # we sum to each batch an offset of batch_num * n_agents to make sure that
    # the adjacency matrices remain independent
    batch_edge_index = edge_index.repeat(1, batch_size) + batch * n_agents
    graphs.edge_index = batch_edge_index

    graphs = graphs.to(x.device)

    return graphs


@dataclass
class GnnConfig(ModelConfig):
    @staticmethod
    def associated_class():
        return Gnn

    @staticmethod
    def get_from_yaml(path: Optional[str] = None) -> GnnConfig:
        if path is None:
            return GnnConfig(
                **ModelConfig._load_from_yaml(
                    name=GnnConfig.associated_class().__name__,
                )
            )
        else:
            return GnnConfig(**parse_model_config(read_yaml_config(path)))
