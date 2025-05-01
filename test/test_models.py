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

from torchrl.data.tensor_specs import Composite, Unbounded


def _get_input_and_output_specs(
    centralised,
    input_has_agent_dim,
    model_name,
    share_params,
    n_agents,
    in_features=2,
    out_features=4,
    x=12,
    y=12,
    set_size=5,
):

    if model_name == "cnn":
        multi_agent_input_shape = (n_agents, x, y, in_features)
        single_agent_input_shape = (x, y, in_features)
    elif model_name == "deepsets":
        multi_agent_input_shape = (n_agents, set_size, in_features)
        single_agent_input_shape = (set_size, in_features)
    else:
        multi_agent_input_shape = (n_agents, in_features)
        single_agent_input_shape = in_features

    other_multi_agent_input_shape = (n_agents, in_features + 1)
    other_single_agent_input_shape = in_features + 1

    if input_has_agent_dim:
        input_spec = Composite(
            {
                "agents": Composite(
                    {
                        "observation": Unbounded(shape=multi_agent_input_shape),
                        "other": Unbounded(shape=other_multi_agent_input_shape),
                    },
                    shape=(n_agents,),
                )
            }
        )
    else:
        input_spec = Composite(
            {
                "observation": Unbounded(shape=single_agent_input_shape),
                "other": Unbounded(shape=other_single_agent_input_shape),
            },
        )

    if output_has_agent_dim(centralised=centralised, share_params=share_params):
        output_spec = Composite(
            {
                "agents": Composite(
                    {"out": Unbounded(shape=(n_agents, out_features))},
                    shape=(n_agents,),
                )
            },
        )
    else:
        output_spec = Composite(
            {"out": Unbounded(shape=(out_features,))},
        )
    return input_spec, output_spec


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
        ["cnn", "gru", "gnn", "mlp"],
        ["cnn", "gru", "mlp"],
        ["gru", "mlp"],
        ["lstm", "gru"],
        ["cnn", "lstm", "mlp"],
        ["lstm", "mlp"],
    ],
)
def test_models_forward_shape(
    share_params, centralised, input_has_agent_dim, model_name, batch_size, n_agents=3
):
    if not input_has_agent_dim and not centralised:
        pytest.skip()  # this combination should never happen
    if ("gnn" in model_name) and (
        not input_has_agent_dim
        or (isinstance(model_name, list) and model_name[0] != "gnn")
    ):
        pytest.skip("gnn model needs agent dim as input")

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

    input_spec, output_spec = _get_input_and_output_specs(
        centralised=centralised,
        input_has_agent_dim=input_has_agent_dim,
        model_name=model_name if isinstance(model_name, str) else model_name[0],
        share_params=share_params,
        n_agents=n_agents,
    )

    if centralised:
        config.is_critic = True
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
    input_td = input_spec.rand()
    if config.is_rnn:
        if len(batch_size) < 2:
            if centralised:
                pytest.skip("rnn model with this batch sizes is a policy")
            hidden_spec = config.get_model_state_spec()
            hidden_spec = Composite(
                {
                    "agents": Composite(
                        hidden_spec.expand(n_agents, *hidden_spec.shape),
                        shape=(n_agents,),
                    )
                }
            )
            input_td.update(hidden_spec.rand())
        input_td["is_init"] = torch.randint(0, 2, (1,), dtype=torch.bool)
    out_td = model(input_td.expand(batch_size))
    assert output_spec.expand(batch_size).is_in(out_td)


@pytest.mark.parametrize("input_has_agent_dim", [True, False])
@pytest.mark.parametrize("centralised", [True, False])
@pytest.mark.parametrize("share_params", [True, False])
@pytest.mark.parametrize(
    "model_name",
    [
        *model_config_registry.keys(),
        ["cnn", "gnn", "mlp"],
        ["cnn", "mlp", "gnn"],
        ["cnn", "mlp"],
        ["cnn", "gru", "gnn", "mlp"],
        ["cnn", "gru", "mlp"],
        ["gru", "mlp"],
        ["lstm", "gru"],
        ["cnn", "lstm", "mlp"],
        ["lstm", "mlp"],
    ],
)
@pytest.mark.parametrize("batch_size", [(), (2,), (3, 2)])
def test_share_params_between_models(
    share_params,
    centralised,
    input_has_agent_dim,
    model_name,
    batch_size,
    n_agents=3,
):
    if not input_has_agent_dim and not centralised:
        pytest.skip()  # this combination should never happen
    if ("gnn" in model_name) and (
        not input_has_agent_dim
        or (isinstance(model_name, list) and model_name[0] != "gnn")
    ):
        pytest.skip("gnn model needs agent dim as input")

    torch.manual_seed(0)

    input_spec, output_spec = _get_input_and_output_specs(
        centralised=centralised,
        input_has_agent_dim=input_has_agent_dim,
        model_name=model_name if isinstance(model_name, str) else model_name[0],
        share_params=share_params,
        n_agents=n_agents,
    )

    if isinstance(model_name, List):
        config = SequenceModelConfig(
            model_configs=[
                model_config_registry[config].get_from_yaml() for config in model_name
            ],
            intermediate_sizes=[4] * (len(model_name) - 1),
        )
    else:
        config = model_config_registry[model_name].get_from_yaml()
    if centralised:
        config.is_critic = True
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
    second_model = config.get_model(
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
    model.share_params_with(second_model)
    for param, second_param in zip(model.parameters(), second_model.parameters()):
        assert torch.eq(param, second_param).all()


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
        input_spec = Composite(
            {
                agent_goup: Composite(
                    {
                        "observation": Unbounded(
                            shape=multi_agent_obs.shape[len(batch_size) :]
                        ),
                        "pos": Unbounded(
                            shape=multi_agent_pos.shape[len(batch_size) :]
                        ),
                    },
                    shape=(n_agents,),
                )
            }
        )

        output_spec = Composite(
            {
                agent_goup: Composite(
                    {"out": Unbounded(shape=(n_agents, out_features))},
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
            exclude_pos_from_node_features=False,
            pos_features=pos_size if position_key is not None else 0,
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
                exclude_pos_from_node_features=False,
                pos_features=pos_size if position_key is not None else 0,
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


class TestDeepsets:
    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("batch_size", [(), (2,), (3, 2)])
    def test_special_case_centralized_critic_from_agent_tensors(
        self,
        share_params,
        batch_size,
        centralised=True,
        input_has_agent_dim=True,
        model_name="deepsets",
        n_agents=3,
        in_features=4,
        out_features=2,
    ):

        torch.manual_seed(0)

        config = model_config_registry[model_name].get_from_yaml()

        multi_agent_input_shape = (n_agents, in_features)
        other_multi_agent_input_shape = (n_agents, in_features)

        input_spec = Composite(
            {
                "agents": Composite(
                    {
                        "observation": Unbounded(shape=multi_agent_input_shape),
                        "other": Unbounded(shape=other_multi_agent_input_shape),
                    },
                    shape=(n_agents,),
                )
            }
        )

        if output_has_agent_dim(centralised=centralised, share_params=share_params):
            output_spec = Composite(
                {
                    "agents": Composite(
                        {"out": Unbounded(shape=(n_agents, out_features))},
                        shape=(n_agents,),
                    )
                },
            )
        else:
            output_spec = Composite(
                {"out": Unbounded(shape=(out_features,))},
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
