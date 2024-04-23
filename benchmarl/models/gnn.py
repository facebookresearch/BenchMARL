#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import importlib
from dataclasses import dataclass, MISSING
from math import prod
from typing import Optional, Type

import torch
from tensordict import TensorDictBase
from torch import nn, Tensor

from benchmarl.models.common import Model, ModelConfig

_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None
if _has_torch_geometric:
    import torch_geometric

TOPOLOGY_TYPES = {"full", "empty"}


def _get_edge_index(topology: str, self_loops: bool, n_agents: int, device: str):
    if topology == "full":
        adjacency = torch.ones(n_agents, n_agents, device=device, dtype=torch.long)
    elif topology == "empty":
        adjacency = torch.ones(n_agents, n_agents, device=device, dtype=torch.long)

    edge_index, _ = torch_geometric.utils.dense_to_sparse(adjacency)

    if self_loops:
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index)
    else:
        edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)

    return edge_index


class Gnn(Model):
    """A GNN model.

    GNN models can be used as "decentralized" actors or critics.

    Args:
        topology (str): Topology of the graph adjacency matrix. Options: "full", "empty".
        self_loops (str): Whether the resulting adjacency matrix will have self loops.
        gnn_class (Type[torch_geometric.nn.MessagePassing]): the gnn convolution class to use
        gnn_kwargs (dict, optional): the dict of arguments to pass to the gnn conv class

    Examples:

        .. code-block:: python

            import torch_geometric
            from torch import nn
            from benchmarl.algorithms import IppoConfig
            from benchmarl.environments import VmasTask
            from benchmarl.experiment import Experiment, ExperimentConfig
            from benchmarl.models import SequenceModelConfig, GnnConfig, MlpConfig
            experiment = Experiment(
                algorithm_config=IppoConfig.get_from_yaml(),
                model_config=GnnConfig(
                    topology="full",
                    self_loops=False,
                    gnn_class=torch_geometric.nn.conv.GATv2Conv,
                    gnn_kwargs={},
                ),
                critic_model_config=SequenceModelConfig(
                    model_configs=[
                        MlpConfig(num_cells=[8], activation_class=nn.Tanh, layer_class=nn.Linear),
                        GnnConfig(
                            topology="full",
                            self_loops=False,
                            gnn_class=torch_geometric.nn.conv.GraphConv,
                        ),
                        MlpConfig(num_cells=[6], activation_class=nn.Tanh, layer_class=nn.Linear),
                    ],
                    intermediate_sizes=[5,3],
                ),
                seed=0,
                config=ExperimentConfig.get_from_yaml(),
                task=VmasTask.NAVIGATION.get_from_yaml(),
            )
            experiment.run()



    """

    def __init__(
        self,
        topology: str,
        self_loops: bool,
        gnn_class: Type[torch_geometric.nn.MessagePassing],
        gnn_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.topology = topology
        self.self_loops = self_loops

        super().__init__(**kwargs)

        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        )
        self.output_features = self.output_leaf_spec.shape[-1]

        if gnn_kwargs is None:
            gnn_kwargs = {}
        gnn_kwargs.update(
            {"in_channels": self.input_features, "out_channels": self.output_features}
        )

        self.gnns = nn.ModuleList(
            [
                gnn_class(**gnn_kwargs).to(self.device)
                for _ in range(self.n_agents if not self.share_params else 1)
            ]
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
        if self.centralised:
            raise ValueError("GNN model can only be used in non-centralised critics")
        if not self.input_has_agent_dim:
            raise ValueError(
                "The GNN module is not compatible with input that does not have the agent dimension,"
                "such as the global state in centralised critics. Please choose another critic model"
                "if your algorithm has a centralized critic and the task has a global state."
            )

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
        input = torch.cat([tensordict.get(in_key) for in_key in self.in_keys], dim=-1)

        batch_size = input.shape[:-2]

        graph = batch_from_dense_to_ptg(x=input, edge_index=self.edge_index)

        if not self.share_params:
            res = torch.stack(
                [
                    gnn(graph.x, graph.edge_index).view(
                        *batch_size,
                        self.n_agents,
                        self.output_features,
                    )[..., i, :]
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


# class GnnKernel(nn.Module):
#     def __init__(self, in_dim, out_dim, **cfg):
#         super().__init__()
#
#         gnn_types = {"GraphConv", "GATv2Conv", "GINEConv"}
#         aggr_types = {"add", "mean", "max"}
#
#         self.aggr = "add"
#         self.gnn_type = "GraphConv"
#
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.activation_fn = nn.Tanh
#
#         if self.gnn_type == "GraphConv":
#             self.gnn = GraphConv(
#                 self.in_dim,
#                 self.out_dim,
#                 aggr=self.aggr,
#             )
#         elif self.gnn_type == "GATv2Conv":
#             # Default adds self loops
#             self.gnn = GATv2Conv(
#                 self.in_dim,
#                 self.out_dim,
#                 edge_dim=self.edge_features,
#                 fill_value=0.0,
#                 share_weights=True,
#                 add_self_loops=True,
#                 aggr=self.aggr,
#             )
#         elif self.gnn_type == "GINEConv":
#             self.gnn = GINEConv(
#                 nn=nn.Sequential(
#                     torch.nn.Linear(self.in_dim, self.out_dim),
#                     self.activation_fn(),
#                 ),
#                 edge_dim=self.edge_features,
#                 aggr=self.aggr,
#             )
#
#     def forward(self, x, edge_index):
#         out = self.gnn(x, edge_index)
#         return out


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
    """Dataclass config for a :class:`~benchmarl.models.Gnn`."""

    topology: str = MISSING
    self_loops: bool = MISSING

    gnn_class: Type[torch_geometric.nn.MessagePassing] = MISSING
    gnn_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return Gnn
