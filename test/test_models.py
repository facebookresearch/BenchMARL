#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from benchmarl.hydra_config import load_model_config_from_hydra
from benchmarl.models import model_config_registry

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


@pytest.mark.parametrize("input_has_agent_dim", [False, True])
@pytest.mark.parametrize("centralised", [True, False])
@pytest.mark.parametrize("share_params", [False, True])
@pytest.mark.parametrize("model_name", model_config_registry.keys())
def test_models_forward_shape(
    share_params, centralised, input_has_agent_dim, model_name
):
    if not input_has_agent_dim and not centralised:
        pytest.skip()  # this combination should never happen
    if model_name == "gnn" and centralised:
        pytest.skip()  # gnn model is always decentralized

    torch.manual_seed(0)
    cnn_config = model_config_registry[model_name].get_from_yaml()
    n_agents = 3
    x = 12
    y = 12
    channels = 3
    out_features = 4

    if model_name == "cnn":
        multi_agent_tensor = torch.rand((n_agents, x, y, channels))
    else:
        multi_agent_tensor = torch.rand((n_agents, channels))
    single_agent_tensor = multi_agent_tensor[0].clone()

    if input_has_agent_dim:
        input_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "observation": UnboundedContinuousTensorSpec(
                            shape=multi_agent_tensor.shape
                        )
                    },
                    shape=(n_agents,),
                )
            }
        )
    else:
        input_spec = CompositeSpec(
            {
                "observation": UnboundedContinuousTensorSpec(
                    shape=single_agent_tensor.shape
                )
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
            }
        )
    else:
        output_spec = CompositeSpec(
            {"out": UnboundedContinuousTensorSpec(shape=(out_features,))},
        )

    cnn_model = cnn_config.get_model(
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
    out_td = cnn_model(input_td)
    assert output_spec.is_in(out_td)
