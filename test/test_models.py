#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest
import torch
from tensordict import TensorDict as td

from benchmarl.hydra_config import (
    load_algorithm_config_from_hydra,
    load_experiment_from_hydra,
    load_model_config_from_hydra,
)
from benchmarl.models import model_config_registry

from benchmarl.models.common import SequenceModelConfig
from hydra import compose, initialize

from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)


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
@pytest.mark.parametrize("model_name", model_config_registry.keys())
def test_mlp_forward_pass(model_name, share_params, centralised, input_has_agent_dim):
    task_name = "vmas/balance"
    algorithm = "mappo"
    with initialize(version_base=None, config_path="../benchmarl/conf"):

        if input_has_agent_dim == False and centralised == False:
            return  # this combination should never happen
        if model_name == "gnn" and centralised == True:
            return  # gnn model is always decentralised

        cfg = compose(
            config_name="config",
            overrides=[
                f"algorithm={algorithm}",
                f"task={task_name}",
                f"model=layers/{model_name}",
                "experiment.loggers=[csv]",
                f"experiment.share_policy_params={share_params}",
                f"algorithm.share_param_critic={share_params}",
            ],
        )
        model_config = load_model_config_from_hydra(cfg.model)

        group = "agents"
        n_agents = 4
        out_dim = 8
        obs_dim = 16

        def test_model(model, expected_shape):
            print("Centralised: ", model.centralised)
            print("Share params: ", model.share_params)
            print("Input has agent dim: ", model.input_has_agent_dim)

            if input_has_agent_dim:
                inputs = td(
                    {"agents": {"observation": torch.randn((1, n_agents, 16))}}
                )  # batch, num_agents, obs_dim
            else:
                inputs = td(
                    {"agents": {"observation": torch.randn((1, 16))}}
                )  # batch, obs_dim
            if model_name == "cnn":
                inputs = td(
                    {
                        "agents": {
                            "observation": torch.randn(
                                [1, n_agents, model_config.in_features, 84, 84]
                            )
                        }
                    }
                )  # batch, obs_dim

            out = model(inputs)

            assert out[model.out_key].shape == expected_shape
            # TODO assert values are different as well to make sure share_params are working properly

        if share_params and centralised:
            expected_out = (
                1,
                out_dim,
            )  # (lost a dimension to centralised. output_has_agent_dim is true when both centralised and share_params is true)
        else:
            expected_out = (1, n_agents, out_dim)

        if input_has_agent_dim:
            output_spec = CompositeSpec(
                {
                    "out": UnboundedContinuousTensorSpec(
                        shape=(
                            n_agents,
                            out_dim,
                        )
                    )
                }
            )
        else:
            output_spec = CompositeSpec(
                {
                    group: CompositeSpec(
                        {
                            "out": UnboundedContinuousTensorSpec(
                                shape=(n_agents, out_dim)
                            )
                        },
                        shape=(n_agents,),
                    )
                }
            )

        obs_dim = (
            n_agents if centralised and model_name == "cnn" else obs_dim
        )  # if centralised and cnn, obs_dim is n_agents as input_features because number of agents rather than actual observation_dim
        input_spec = CompositeSpec(
            {
                group: CompositeSpec(
                    {
                        "observation": UnboundedContinuousTensorSpec(
                            shape=(n_agents, obs_dim)
                        ).clone()
                    },
                    shape=(n_agents,),
                )
            }
        )

        action_spec = CompositeSpec(
            {
                group: CompositeSpec(
                    {
                        "action": BoundedTensorSpec(
                            shape=torch.Size([4, 2]),
                            low=torch.full((4, 2), -1.0),
                            high=torch.full((4, 2), 1.0),
                        )
                    },
                    shape=torch.Size([4]),
                )
            }
        )

        model = model_config.get_model(
            input_spec=input_spec,
            output_spec=output_spec,
            n_agents=n_agents,
            centralised=centralised,
            input_has_agent_dim=input_has_agent_dim,
            agent_group="agents",
            share_params=share_params,
            device="cpu",
            action_spec=action_spec,
        )

        test_model(model, expected_out)
