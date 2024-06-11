#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import contextlib
from typing import List

import pytest
import torch
import torch_geometric.nn

from benchmarl.hydra_config import load_model_config_from_hydra
from benchmarl.models import GnnConfig, model_config_registry

from benchmarl.models.common import output_has_agent_dim, SequenceModelConfig
from hydra import compose, initialize

from torchrl.data.tensor_specs import CompositeSpec, UnboundedContinuousTensorSpec


@pytest.mark.parametrize("model_name", model_config_registry.keys())
def test_loading_simple_models(model_name):
    with initialize(version_base=None, config_path="../benchmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=mappo",
                "task=vmas/balance",
                f"model=layers/{model_name}",
            ],
        )
        model_config = load_model_config_from_hydra(cfg.model)
        assert model_config == model_config_registry[model_name].get_from_yaml()


@pytest.mark.parametrize("model_name", model_config_registry.keys())
def test_loading_sequence_models(model_name, intermediate_size=10):
    with initialize(version_base=None, config_path="../benchmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=mappo",
                "task=vmas/balance",
                "model=sequence",
                f"model/layers@model.layers.l1={model_name}",
                f"model/layers@model.layers.l2={model_name}",
                f"+model/layers@model.layers.l3={model_name}",
                f"model.intermediate_sizes={[intermediate_size,intermediate_size]}",
            ],
        )
        hydra_model_config = load_model_config_from_hydra(cfg.model)
        layer_config = model_config_registry[model_name].get_from_yaml()
        yaml_config = SequenceModelConfig(
            model_configs=[layer_config, layer_config, layer_config],
            intermediate_sizes=[intermediate_size, intermediate_size],
        )
        assert hydra_model_config == yaml_config


@pytest.mark.parametrize("input_has_agent_dim", [True, False])
@pytest.mark.parametrize("centralised", [True, False])
@pytest.mark.parametrize("share_params", [True, False])
@pytest.mark.parametrize("batch_size", [(), (2,), (3, 2)])
@pytest.mark.parametrize(
    "model_name",
    [
        *model_config_registry.keys(),
        ["cnn", "gnn", "mlp"],
        ["cnn", "mlp", "gnn"],
        ["cnn", "mlp"],
    ],
)
def test_models_forward_shape(
    share_params, centralised, input_has_agent_dim, model_name, batch_size
):
    if not input_has_agent_dim and not centralised:
        pytest.skip()  # this combination should never happen
    if ("gnn" in model_name) and centralised:
        pytest.skip("gnn model is always decentralized")

    torch.manual_seed(0)

    if isinstance(model_name, List):
        config = SequenceModelConfig(
            model_configs=[
                model_config_registry[config].get_from_yaml() for config in model_name
            ],
            intermediate_sizes=[4] * (len(model_name) - 1),
        )
    else:
        config = model_config_registry[model_name].get_from_yaml()

    n_agents = 2
    x = 12
    y = 12
    channels = 3
    out_features = 4

    if "cnn" in model_name:
        multi_agent_tensor = torch.rand((*batch_size, n_agents, x, y, channels))
        single_agent_tensor = torch.rand((*batch_size, x, y, channels))
    else:
        multi_agent_tensor = torch.rand((*batch_size, n_agents, channels))
        single_agent_tensor = torch.rand((*batch_size, channels))

    other_multi_agent_tensor = torch.rand((*batch_size, n_agents, channels))
    other_single_agent_tensor = torch.rand((*batch_size, channels))

    if input_has_agent_dim:
        input_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "observation": UnboundedContinuousTensorSpec(
                            shape=multi_agent_tensor.shape[len(batch_size) :]
                        ),
                        "other": UnboundedContinuousTensorSpec(
                            shape=other_multi_agent_tensor.shape[len(batch_size) :]
                        ),
                    },
                    shape=(n_agents,),
                )
            }
        )
    else:
        input_spec = CompositeSpec(
            {
                "observation": UnboundedContinuousTensorSpec(
                    shape=single_agent_tensor.shape[len(batch_size) :]
                ),
                "other": UnboundedContinuousTensorSpec(
                    shape=other_single_agent_tensor.shape[len(batch_size) :]
                ),
            },
        )

    if output_has_agent_dim(centralised=centralised, share_params=share_params):
        output_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "out": UnboundedContinuousTensorSpec(
                            shape=(n_agents, out_features)
                        )
                    },
                    shape=(n_agents,),
                )
            },
        )
    else:
        output_spec = CompositeSpec(
            {"out": UnboundedContinuousTensorSpec(shape=(out_features,))},
        )

    model = config.get_model(
        input_spec=input_spec,
        output_spec=output_spec,
        share_params=share_params,
        centralised=centralised,
        input_has_agent_dim=input_has_agent_dim,
        n_agents=n_agents,
        device="cpu",
        agent_group="agents",
        action_spec=None,
    )
    input_td = input_spec.expand(batch_size).rand()
    out_td = model(input_td)
    assert output_spec.expand(batch_size).is_in(out_td)


class TestGnn:
    @pytest.mark.parametrize("batch_size", [(), (2,), (3, 2)])
    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("position_key", ["pos", None])
    def test_gnn_edge_attrs(
        self,
        batch_size,
        share_params,
        position_key,
        n_agents=3,
        obs_size=4,
        pos_size=2,
        agent_goup="agents",
        out_features=5,
    ):
        torch.manual_seed(0)

        multi_agent_obs = torch.rand((*batch_size, n_agents, obs_size))
        multi_agent_pos = torch.rand((*batch_size, n_agents, pos_size))
        input_spec = CompositeSpec(
            {
                agent_goup: CompositeSpec(
                    {
                        "observation": UnboundedContinuousTensorSpec(
                            shape=multi_agent_obs.shape[len(batch_size) :]
                        ),
                        "pos": UnboundedContinuousTensorSpec(
                            shape=multi_agent_obs.shape[len(batch_size) :]
                        ),
                    },
                    shape=(n_agents,),
                )
            }
        )

        output_spec = CompositeSpec(
            {
                agent_goup: CompositeSpec(
                    {
                        "out": UnboundedContinuousTensorSpec(
                            shape=(n_agents, out_features)
                        )
                    },
                    shape=(n_agents,),
                )
            },
        )

        # Test with correct stuff
        gnn = GnnConfig(
            topology="full",
            self_loops=True,
            gnn_class=torch_geometric.nn.GATv2Conv,
            gnn_kwargs=None,
            position_key=position_key,
        ).get_model(
            input_spec=input_spec,
            output_spec=output_spec,
            agent_group=agent_goup,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=share_params,
            device="cpu",
            action_spec=None,
        )

        obs_input = input_spec.expand(batch_size).rand()
        output = gnn(obs_input)
        assert output_spec.expand(batch_size).is_in(output)

        # Test with a GNN without edge_attrs
        with (
            pytest.warns(
                match="Position key or velocity key provided but GNN class does not support edge attributes*"
            )
            if position_key is not None
            else contextlib.nullcontext()
        ):
            gnn = GnnConfig(
                topology="full",
                self_loops=True,
                gnn_class=torch_geometric.nn.GraphConv,
                gnn_kwargs=None,
                position_key=position_key,
            ).get_model(
                input_spec=input_spec,
                output_spec=output_spec,
                agent_group=agent_goup,
                input_has_agent_dim=True,
                n_agents=n_agents,
                centralised=False,
                share_params=share_params,
                device="cpu",
                action_spec=None,
            )
        obs_input = input_spec.expand(batch_size).rand()
        output = gnn(obs_input)
        assert output_spec.expand(batch_size).is_in(output)
