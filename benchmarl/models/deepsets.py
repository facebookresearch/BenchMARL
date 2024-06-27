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
from torch import nn, Tensor
from torchrl.modules import MLP

from benchmarl.models.common import Model, ModelConfig


class Deepsets(Model):
    r"""Deepsets Model from `this paper <https://arxiv.org/abs/1703.06114>`__ .

    The BenchMARL Deepsets accepts multiple inputs of 2 types:

    - sets :math:`s` : Tensors of shape ``(*batch,S,F)``
    - arrays :math:`x` : Tensors of shape ``(*batch,F)``

    The Deepsets model will check that all set inputs have the same shape (excluding the last dimension)
    and cat them along that dimension before processing them.

    It will check that all array inputs have the same shape (excluding the last dimension)
    and cat them along that dimension.

    It will then compute the output according to the following function.

    .. math::

       \rho \left (x, \bigoplus_{s\in S}\phi(s) \right ),

    where :math:`\rho,\phi` are MLPs configurable in the model setup.

    The model is useful in various contexts, for example:

    - When used as a policy (``self.centralized==False``, ``self.input_has_agent_dim==True``), it can process
      observations with shape ``(*batch,n_agents,S,F)``, reducing them to ``(*batch,n_agents,F)``
    - When used a a centralized crtic with a global state as input
      (``self.centralized==True``, ``self.input_has_agent_dim==False``), it can process the global state with shape
      ``(*batch,S,F)`` , reducing it to ``(*batch,F)``.
    - When used a a centralized crtic with local agent observations as input
      (``self.centralized==True``, ``self.input_has_agent_dim==True``), it can process normal agent observations with shape
      ``(*batch,n_agents,F)``, reducing them to ``(*batch,F)``. **Note**: If the agents also have set observations
      ``(*batch,n_agents,S,F)`` it will apply two deep sets networks. The first will remove the set dimension
      in the agents' inputs (``(*batch,n_agents,F)``), and the second will remove the agent dimension  (``(*batch,F)``).
      Both networks will share the same configuration.

    Args:
        aggr (str): The aggregation strategy to use in the Deepsets model.
        local_nn_num_cells (Sequence[int]): number of cells of every layer in between the input and output in the :math:`\phi` MLP.
        local_nn_activation_class (Type[nn.Module]): activation class to be used in the :math:`\phi` MLP.
        out_features_local_nn (int): output features of the :math:`\phi` MLP.
        global_nn_num_cells (Sequence[int]): number of cells of every layer in between the input and output in the :math:`\rho` MLP.
        global_nn_activation_class (Type[nn.Module]): activation class to be used in the :math:`\rho` MLP.


    """

    def __init__(
        self,
        aggr: str,
        local_nn_num_cells: Sequence[int],
        local_nn_activation_class: Type[nn.Module],
        out_features_local_nn: int,
        global_nn_num_cells: Sequence[int],
        global_nn_activation_class: Type[nn.Module],
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.aggr = aggr
        self.local_nn_num_cells = local_nn_num_cells
        self.local_nn_activation_class = local_nn_activation_class
        self.global_nn_num_cells = global_nn_num_cells
        self.global_nn_activation_class = global_nn_activation_class
        self.out_features_local_nn = out_features_local_nn

        self.input_local_set_features = sum(
            [self.input_spec[key].shape[-1] for key in self.set_in_keys_local]
        )
        self.input_local_tensor_features = sum(
            [self.input_spec[key].shape[-1] for key in self.tensor_in_keys_local]
        )
        self.input_global_set_features = sum(
            [self.input_spec[key].shape[-1] for key in self.set_in_keys_global]
        )
        self.input_global_tensor_features = sum(
            [self.input_spec[key].shape[-1] for key in self.tensor_in_keys_global]
        )

        self.output_features = self.output_leaf_spec.shape[-1]

        if self.input_local_set_features > 0:  # Need local deepsets
            self.local_deepsets = nn.ModuleList(
                [
                    self._make_deepsets_net(
                        in_features=self.input_local_set_features,
                        out_features_local_nn=self.out_features_local_nn,
                        in_fetures_global_nn=self.out_features_local_nn
                        + self.input_local_tensor_features,
                        out_features=(
                            self.output_features
                            if not self.centralised
                            else self.out_features_local_nn
                        ),
                        aggr=self.aggr,
                        local_nn_activation_class=self.local_nn_activation_class,
                        global_nn_activation_class=self.global_nn_activation_class,
                        local_nn_num_cells=self.local_nn_num_cells,
                        global_nn_num_cells=self.global_nn_num_cells,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )
        if self.centralised:  # Need global deepsets
            self.global_deepsets = nn.ModuleList(
                [
                    self._make_deepsets_net(
                        in_features=(
                            self.input_global_set_features
                            if self.input_local_set_features == 0
                            else self.out_features_local_nn
                        ),
                        out_features_local_nn=self.out_features_local_nn,
                        in_fetures_global_nn=self.out_features_local_nn
                        + self.input_global_tensor_features,
                        out_features=self.output_features,
                        aggr=self.aggr,
                        local_nn_activation_class=self.local_nn_activation_class,
                        global_nn_activation_class=self.global_nn_activation_class,
                        local_nn_num_cells=self.local_nn_num_cells,
                        global_nn_num_cells=self.global_nn_num_cells,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

    def _make_deepsets_net(
        self,
        in_features: int,
        out_features: int,
        aggr: str,
        local_nn_num_cells: Sequence[int],
        local_nn_activation_class: Type[nn.Module],
        global_nn_num_cells: Sequence[int],
        global_nn_activation_class: Type[nn.Module],
        out_features_local_nn: int,
        in_fetures_global_nn: int,
    ) -> _DeepsetsNet:
        local_nn = MLP(
            in_features=in_features,
            out_features=out_features_local_nn,
            num_cells=local_nn_num_cells,
            activation_class=local_nn_activation_class,
            device=self.device,
        )
        global_nn = MLP(
            in_features=in_fetures_global_nn,
            out_features=out_features,
            num_cells=global_nn_num_cells,
            activation_class=global_nn_activation_class,
            device=self.device,
        )
        return _DeepsetsNet(local_nn, global_nn, aggr=aggr)

    def _perform_checks(self):
        super()._perform_checks()

        input_shape_tensor_local = None
        self.tensor_in_keys_local = []
        input_shape_set_local = None
        self.set_in_keys_local = []

        input_shape_tensor_global = None
        self.tensor_in_keys_global = []
        input_shape_set_global = None
        self.set_in_keys_global = []

        error_invalid_input = ValueError(
            f"DeepSet set inputs should all have the same shape up to the last dimension, got {self.input_spec}"
        )

        for input_key, input_spec in self.input_spec.items(True, True):
            if self.input_has_agent_dim and len(input_spec.shape) == 3:
                self.set_in_keys_local.append(input_key)
                if input_shape_set_local is None:
                    input_shape_set_local = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_set_local:
                    raise error_invalid_input
            elif self.input_has_agent_dim and len(input_spec.shape) == 2:
                self.tensor_in_keys_local.append(input_key)
                if input_shape_tensor_local is None:
                    input_shape_tensor_local = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_tensor_local:
                    raise error_invalid_input
            elif not self.input_has_agent_dim and len(input_spec.shape) == 2:
                self.set_in_keys_global.append(input_key)
                if input_shape_set_global is None:
                    input_shape_set_global = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_set_global:
                    raise error_invalid_input
            elif not self.input_has_agent_dim and len(input_spec.shape) == 1:
                self.tensor_in_keys_global.append(input_key)
                if input_shape_tensor_global is None:
                    input_shape_tensor_global = input_spec.shape[:-1]
                elif input_spec.shape[:-1] != input_shape_tensor_global:
                    raise error_invalid_input
            else:
                raise ValueError(
                    f"DeepSets input value {input_key} from {self.input_spec} has an invalid shape"
                )

        # Centralized model not needing any local deepsets
        if (
            self.centralised
            and not len(self.set_in_keys_local)
            and self.input_has_agent_dim
        ):
            self.set_in_keys_global = self.tensor_in_keys_local
            input_shape_set_global = input_shape_tensor_local
            self.tensor_in_keys_local = []

        if (not self.centralised and not len(self.set_in_keys_local)) or (
            self.centralised
            and not self.input_has_agent_dim
            and not len(self.set_in_keys_global)
        ):
            raise ValueError("DeepSets found no set inputs, maybe use an MLP?")

        if len(self.set_in_keys_local) and input_shape_set_local[-2] != self.n_agents:
            raise ValueError()
        if (
            len(self.tensor_in_keys_local)
            and input_shape_tensor_local[-1] != self.n_agents
        ):
            raise ValueError()
        if (
            len(self.set_in_keys_global)
            and self.input_has_agent_dim
            and input_shape_set_global[-1] != self.n_agents
        ):
            raise ValueError()

        if (
            self.output_has_agent_dim
            and (
                self.output_leaf_spec.shape[-2] != self.n_agents
                or len(self.output_leaf_spec.shape) != 2
            )
        ) or (not self.output_has_agent_dim and len(self.output_leaf_spec.shape) != 1):
            raise ValueError()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if len(self.set_in_keys_local):
            # Local deep sets
            input_local_sets = torch.cat(
                [tensordict.get(in_key) for in_key in self.set_in_keys_local], dim=-1
            )
            input_local_tensors = None
            if len(self.tensor_in_keys_local):
                input_local_tensors = torch.cat(
                    [tensordict.get(in_key) for in_key in self.tensor_in_keys_local],
                    dim=-1,
                )
            if self.share_params:
                local_output = self.local_deepsets[0](
                    input_local_sets, input_local_tensors
                )
            else:
                local_output = torch.stack(
                    [
                        net(input_local_sets, input_local_tensors)[..., i, :]
                        for i, net in enumerate(self.local_deepsets)
                    ],
                    dim=-2,
                )
        else:
            local_output = None

        if self.centralised:
            if local_output is None:
                # gather local output
                local_output = torch.cat(
                    [tensordict.get(in_key) for in_key in self.set_in_keys_global],
                    dim=-1,
                )
            # Global deepsets
            input_global_tensors = None
            if len(self.tensor_in_keys_global):
                input_global_tensors = torch.cat(
                    [tensordict.get(in_key) for in_key in self.tensor_in_keys_global],
                    dim=-1,
                )
            if self.share_params:
                global_output = self.global_deepsets[0](
                    local_output, input_global_tensors
                )
            else:
                global_output = torch.stack(
                    [
                        net(local_output, input_global_tensors)
                        for i, net in enumerate(self.global_deepsets)
                    ],
                    dim=-2,
                )
            tensordict.set(self.out_key, global_output)
        else:
            tensordict.set(self.out_key, local_output)

        return tensordict


class _DeepsetsNet(nn.Module):
    """https://arxiv.org/abs/1703.06114"""

    def __init__(
        self,
        local_nn: torch.nn.Module,
        global_nn: torch.nn.Module,
        set_dim: int = -2,
        aggr: str = "sum",
    ):
        super().__init__()
        self.aggr = aggr
        self.set_dim = set_dim
        self.local_nn = local_nn
        self.global_nn = global_nn

    def forward(self, x: Tensor, extra_global_input: Optional[Tensor]) -> Tensor:
        x = self.local_nn(x)
        x = self.reduce(x, dim=self.set_dim, aggr=self.aggr)
        if extra_global_input is not None:
            x = torch.cat([x, extra_global_input], dim=-1)
        x = self.global_nn(x)
        return x

    @staticmethod
    def reduce(x: Tensor, dim: int, aggr: str) -> Tensor:
        if aggr == "sum" or aggr == "add":
            return torch.sum(x, dim=dim)
        elif aggr == "mean":
            return torch.mean(x, dim=dim)
        elif aggr == "max":
            return torch.max(x, dim=dim)[0]
        elif aggr == "min":
            return torch.min(x, dim=dim)[0]
        elif aggr == "mul":
            return torch.prod(x, dim=dim)


@dataclass
class DeepsetsConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Deepsets`."""

    aggr: str = MISSING
    out_features_local_nn: int = MISSING

    local_nn_num_cells: Sequence[int] = MISSING
    local_nn_activation_class: Type[nn.Module] = MISSING

    global_nn_num_cells: Sequence[int] = MISSING
    global_nn_activation_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        return Deepsets
