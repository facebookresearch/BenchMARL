#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import importlib
import inspect
import warnings
from dataclasses import dataclass, MISSING
from math import prod
from typing import List, Optional, Type

import torch
from tensordict import TensorDictBase
from tensordict.utils import _unravel_key_to_tuple, NestedKey
from torch import nn, Tensor

from benchmarl.models.common import Model, ModelConfig

_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None
if _has_torch_geometric:
    import torch_geometric
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


class Gnn(Model):
    """A GNN model.

    GNN models can be used as "decentralized" actors or critics.

    Args:
        topology (str): Topology of the graph adjacency matrix. Options: "full", "empty", "from_pos". "from_pos" builds
            the topology dynamically based on ``position_key`` and ``edge_radius``.
        self_loops (str): Whether the resulting adjacency matrix will have self loops.
        gnn_class (Type[torch_geometric.nn.MessagePassing]): the gnn convolution class to use
        gnn_kwargs (dict, optional): the dict of arguments to pass to the gnn conv class
        position_key (str, optional): if provided, it will need to match a leaf key in the tensordict coming from the env
            (in the `observation_spec`) representing the agent position.
            To do this, your environment needs to have dictionary observations and one of the keys needs to be `position_key`.
            This key will be processed as a node feature (unless exclude_pos_from_node_features=True) and it will be used to construct edge features.
            In particular, it will be used to compute relative positions (``pos_node_1 - pos_node_2``) and a
            one-dimensional distance for all neighbours in the graph.
            If you want to use this feature in a :class:`~benchmarl.models.SequenceModel`, the GNN needs to be first in sequence.
        pos_features (int, optional): Needed when position_key is specified.
            It has to match to the last element of the shape the tensor under position_key.
        exclude_pos_from_node_features (optional, bool): If ``position_key`` is provided,
            wether to use it just to compute edge features or also include it in node features.
        velocity_key (str, optional): if provided, it will need to match a leaf key in the tensordict coming from the env
            (in the `observation_spec`) representing the agent position.
            To do this, your environment needs to have dictionary observations and one of the keys needs to be `velocity_key`.
            This key will be processed as a node feature, and
            it will be used to construct edge features. In particular, it will be used to compute relative velocities
            (``vel_node_1 - vel_node_2``) for all neighbours in the graph.
            If you want to use this feature in a :class:`~benchmarl.models.SequenceModel`, the GNN needs to be first in sequence.
        vel_features (int, optional): Needed when velocity_key is specified.
            It has to match to the last element of the shape the tensor under velocity_key.
        edge_radius (float, optional): If topology is ``"from_pos"`` the radius to use to build the agent graph.
            Agents within this radius distance will be neighnours.

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
        gnn_kwargs: Optional[dict],
        position_key: Optional[str],
        exclude_pos_from_node_features: Optional[bool],
        velocity_key: Optional[str],
        edge_radius: Optional[float],
        pos_features: Optional[int],
        vel_features: Optional[int],
        **kwargs,
    ):
        self.topology = topology
        self.self_loops = self_loops
        self.position_key = position_key
        self.velocity_key = velocity_key
        self.exclude_pos_from_node_features = exclude_pos_from_node_features
        self.edge_radius = edge_radius
        self.pos_features = pos_features
        self.vel_features = vel_features

        super().__init__(**kwargs)

        if self.pos_features > 0:
            self.pos_features += 1  # We will add also 1-dimensional distance
        self.edge_features = self.pos_features + self.vel_features
        self.input_features = sum(
            [
                spec.shape[-1]
                for key, spec in self.input_spec.items(True, True)
                if _unravel_key_to_tuple(key)[-1] not in (position_key, velocity_key)
            ]
        )  # Input keys
        if self.position_key is not None and not self.exclude_pos_from_node_features:
            self.input_features += self.pos_features - 1
        if self.velocity_key is not None:
            self.input_features += self.vel_features

        self.output_features = self.output_leaf_spec.shape[-1]

        if gnn_kwargs is None:
            gnn_kwargs = {}
        gnn_kwargs.update(
            {"in_channels": self.input_features, "out_channels": self.output_features}
        )
        self.gnn_supports_edge_attrs = (
            "edge_dim" in inspect.getfullargspec(gnn_class).args
        )
        if (
            self.position_key is not None or self.velocity_key is not None
        ) and not self.gnn_supports_edge_attrs:
            warnings.warn(
                "Position key or velocity key provided but GNN class does not support edge attributes. "
                "These keys will not be used for computing edge features."
            )
        if (
            position_key is not None or velocity_key is not None
        ) and self.gnn_supports_edge_attrs:
            gnn_kwargs.update({"edge_dim": self.edge_features})

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
        self._full_position_key = None
        self._full_velocity_key = None

    def _perform_checks(self):
        super()._perform_checks()

        if self.topology not in TOPOLOGY_TYPES:
            raise ValueError(
                f"Got topology: {self.topology} but only available options are {TOPOLOGY_TYPES}"
            )
        if self.topology == "from_pos" and self.position_key is None:
            raise ValueError("If topology is from_pos, position_key must be provided")
        if (
            self.position_key is not None
            and self.exclude_pos_from_node_features is None
        ):
            raise ValueError(
                "exclude_pos_from_node_features needs to be specified when position_key is provided"
            )
        if self.position_key is not None and self.pos_features <= 0:
            raise ValueError(
                f"Position key specified but pos_features is {self.pos_features}"
            )
        elif self.position_key is None and self.pos_features > 0:
            raise ValueError(
                f"If no position_key is given, pos_features needs to be 0, got: {self.pos_features}"
            )
        if self.velocity_key is not None and self.vel_features <= 0:
            raise ValueError(
                f"Velocity key specified but vel_features is {self.vel_features}"
            )
        elif self.velocity_key is None and self.vel_features > 0:
            raise ValueError(
                f"If no velocity_key is given, vel_features needs to be 0, got: {self.vel_features}"
            )

        if not self.input_has_agent_dim:
            raise ValueError(
                "The GNN module is not compatible with input that does not have the agent dimension,"
                "such as the global state in centralised critics. Please choose another critic model"
                "if your algorithm has a centralized critic and the task has a global state."
                "If you are using the GNN in a centralized critic, it should be the first layer."
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
        input = [
            tensordict.get(in_key)
            for in_key in self.in_keys
            if _unravel_key_to_tuple(in_key)[-1]
            not in (self.position_key, self.velocity_key)
        ]

        # Retrieve position
        if self.position_key is not None:
            if self._full_position_key is None:  # Run once to find full key
                self._full_position_key = self._get_key_terminating_with(
                    list(tensordict.keys(True, True)), self.position_key
                )
                pos = tensordict.get(self._full_position_key)
                if pos.shape[-1] != self.pos_features - 1:
                    raise ValueError(
                        f"Position key in tensordict is {pos.shape[-1]}-dimensional, "
                        f"while model was configured with pos_features={self.pos_features-1}"
                    )
            else:
                pos = tensordict.get(self._full_position_key)
            if not self.exclude_pos_from_node_features:
                input.append(pos)
        else:
            pos = None

        # Retrieve velocity
        if self.velocity_key is not None:
            if self._full_velocity_key is None:  # Run once to find full key
                self._full_velocity_key = self._get_key_terminating_with(
                    list(tensordict.keys(True, True)), self.velocity_key
                )
                vel = tensordict.get(self._full_velocity_key)
                if vel.shape[-1] != self.vel_features:
                    raise ValueError(
                        f"Velocity key in tensordict is {vel.shape[-1]}-dimensional, "
                        f"while model was configured with vel_features={self.vel_features}"
                    )
            else:
                vel = tensordict.get(self._full_velocity_key)
            input.append(vel)
        else:
            vel = None

        input = torch.cat(input, dim=-1)
        batch_size = input.shape[:-2]

        graph = _batch_from_dense_to_ptg(
            x=input,
            edge_index=self.edge_index,
            pos=pos,
            vel=vel,
            self_loops=self.self_loops,
            edge_radius=self.edge_radius,
        )
        forward_gnn_params = {
            "x": graph.x,
            "edge_index": graph.edge_index,
        }
        if (
            self.position_key is not None or self.velocity_key is not None
        ) and self.gnn_supports_edge_attrs:
            forward_gnn_params.update({"edge_attr": graph.edge_attr})

        if not self.share_params:
            if not self.centralised:
                res = torch.stack(
                    [
                        gnn(**forward_gnn_params).view(
                            *batch_size,
                            self.n_agents,
                            self.output_features,
                        )[..., i, :]
                        for i, gnn in enumerate(self.gnns)
                    ],
                    dim=-2,
                )
            else:
                res = torch.stack(
                    [
                        gnn(**forward_gnn_params)
                        .view(
                            *batch_size,
                            self.n_agents,
                            self.output_features,
                        )
                        .mean(dim=-2)  # Mean pooling
                        for i, gnn in enumerate(self.gnns)
                    ],
                    dim=-2,
                )

        else:
            res = self.gnns[0](**forward_gnn_params).view(
                *batch_size, self.n_agents, self.output_features
            )
            if self.centralised:
                res = res.mean(dim=-2)  # Mean pooling

        tensordict.set(self.out_key, res)
        return tensordict

    def _get_key_terminating_with(self, keys: List[NestedKey], key: str) -> NestedKey:
        for k in keys:
            k_tuple = _unravel_key_to_tuple(k)
            if (
                k_tuple[-1] == key
                and self.agent_group in k_tuple
                and not "next" == k_tuple[0]
            ):
                return k
        raise KeyError(
            f"Key terminating with {key} and containing {self.agent_group} not found in keys: {keys}. "
            f"If you are using the GNN in a `SequenceModel` and want to use this key, it needs to be the first model."
        )


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
class GnnConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Gnn`."""

    topology: str = MISSING
    self_loops: bool = MISSING

    gnn_class: Type[torch_geometric.nn.MessagePassing] = MISSING
    gnn_kwargs: Optional[dict] = None

    position_key: Optional[str] = None
    pos_features: Optional[int] = 0
    velocity_key: Optional[str] = None
    vel_features: Optional[int] = 0
    exclude_pos_from_node_features: Optional[bool] = None
    edge_radius: Optional[float] = None

    @staticmethod
    def associated_class():
        return Gnn
