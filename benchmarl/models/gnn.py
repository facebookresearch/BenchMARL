from __future__ import annotations

from dataclasses import dataclass, MISSING

from typing import Optional, Sequence, Type

import torch
import torch_geometric
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from tensordict import TensorDictBase
from torch import nn, Tensor
from torch_geometric.nn import GATv2Conv, GINEConv, GraphConv, MessagePassing
from torch_geometric.transforms import BaseTransform
from torchrl.modules import MLP, MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig, parse_model_config
from benchmarl.utils import read_yaml_config


class Gnn(Model):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        if self.input_has_agent_dim:
            self.mlp = MultiAgentMLP(
                n_agent_inputs=self.input_features,
                n_agent_outputs=self.output_features,
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                **kwargs,
            )
        else:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_features=self.input_features,
                        out_features=self.output_features,
                        device=self.device,
                        **kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

    def _perform_checks(self):
        super()._perform_checks()

        if self.input_has_agent_dim and self.input_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError(
                "If the MLP input has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the MLP output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        input = tensordict.get(self.in_key)

        # Has multi-agent input dimension
        if self.input_has_agent_dim:
            res = self.mlp.forward(input)
            if not self.output_has_agent_dim:
                # If we are here the module is centralised and parameter shared.
                # Thus the multi-agent dimension has been expanded,
                # We remove it without loss of data
                res = res[..., 0, :]

        # Does not have multi-agent input dimension
        else:
            if not self.share_params:
                res = torch.stack(
                    [net(input) for net in self.mlp],
                    dim=-2,
                )
            else:
                res = self.mlp[0](input)

        tensordict.set(self.out_key, res)
        return tensordict


class GnnKernel(nn.Module):
    def __init__(self, in_dim, out_dim, edge_features, **cfg):
        super().__init__()

        gnn_types = {"GraphConv", "GATv2Conv", "GINEConv"}
        aggr_types = {"add", "mean", "max"}

        self.aggr = cfg["aggr"]
        self.gnn_type = cfg["gnn_type"]

        assert self.aggr in aggr_types
        assert self.gnn_type in gnn_types

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_features = edge_features
        self.activation_fn = get_activation_fn(cfg["activation_fn"])

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
        elif self.gnn_type == "MatPosConv":
            self.gnn = MatPosConv(
                self.in_dim,
                self.out_dim,
                edge_features=self.edge_features,
                **cfg,
            )
        else:
            assert False

    def forward(self, x, edge_index, edge_attr):
        if self.gnn_type == "GraphConv":
            out = self.gnn(x, edge_index)
        elif (
            self.gnn_type == "GATv2Conv"
            or self.gnn_type == "GINEConv"
            or self.gnn_type == "MatPosConv"
        ):
            out = self.gnn(x, edge_index, edge_attr)
        else:
            assert False

        return out


def batch_from_dense_to_ptg(
    x,
    pos: Tensor = None,
    vel: Tensor = None,
    edge_index: Tensor = None,
    comm_radius: float = -1,
    rel_pos: bool = True,
    distance: bool = True,
    rel_vel: bool = True,
) -> torch_geometric.data.Batch:
    batch_size = x.shape[0]
    n_agents = x.shape[1]

    x = x.view(-1, x.shape[-1])
    if pos is not None:
        pos = pos.view(-1, pos.shape[-1])
    if vel is not None:
        vel = vel.view(-1, vel.shape[-1])

    assert (edge_index is None or comm_radius < 0) and (
        edge_index is not None or comm_radius > 0
    )

    b = torch.arange(batch_size, device=x.device)

    graphs = torch_geometric.data.Batch()
    graphs.ptr = torch.arange(0, (batch_size + 1) * n_agents, n_agents)
    graphs.batch = torch.repeat_interleave(b, n_agents)
    graphs.pos = pos
    graphs.vel = vel
    graphs.x = x
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
        assert pos is not None
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(
            graphs.pos, batch=graphs.batch, r=comm_radius, loop=False
        )

    graphs = graphs.to(x.device)

    if pos is not None and rel_pos:
        graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs)
    if pos is not None and distance:
        graphs = torch_geometric.transforms.Distance(norm=False)(graphs)
    if vel is not None and rel_vel:
        graphs = RelVel()(graphs)

    return graphs


@dataclass
class GnnConfig(ModelConfig):
    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    activation_class: Type[nn.Module] = MISSING
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return Mlp

    @staticmethod
    def get_from_yaml(path: Optional[str] = None) -> MlpConfig:
        if path is None:
            return MlpConfig(
                **ModelConfig._load_from_yaml(
                    name=MlpConfig.associated_class().__name__,
                )
            )
        else:
            return MlpConfig(**parse_model_config(read_yaml_config(path)))
