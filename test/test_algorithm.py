#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import copy
import warnings
from dataclasses import asdict

import pytest
import torch
from benchmarl.algorithms import (
    algorithm_config_registry,
    Ippo,
    IppoConfig,
    Mappo,
    MappoConfig,
)
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.hydra_config import load_algorithm_config_from_hydra
from benchmarl.models import MlpConfig
from hydra import compose, initialize
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.data.tensor_specs import Categorical, Composite, OneHot, Unbounded
from torchrl.objectives import ClipPPOLoss


@pytest.mark.parametrize("algo_name", algorithm_config_registry.keys())
def test_loading_algorithms(algo_name):
    with initialize(version_base=None, config_path="../benchmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"algorithm={algo_name}",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert algo_config == algorithm_config_registry[algo_name].get_from_yaml()


def _make_experiment(stub_algo_config):
    from types import SimpleNamespace

    model_config = MlpConfig(num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear)
    critic_model_config = copy.deepcopy(model_config)
    critic_model_config.is_critic = True

    return SimpleNamespace(
        config=SimpleNamespace(
            train_device="cpu",
            buffer_device="cpu",
            gamma=0.99,
            share_policy_params=True,
        ),
        model_config=model_config,
        critic_model_config=critic_model_config,
        on_policy=True,
        group_map={"agents": [0, 1]},
        observation_spec=Composite({"agents": Composite({"observation": Unbounded(shape=(2, 3))}, shape=(2,))}),
        action_spec=Composite({"agents": Composite({"action": Categorical(shape=(2,), n=2)}, shape=(2,))}),
        state_spec=None,
        action_mask_spec=None,
        algorithm_config=stub_algo_config,
    )

def _make_ppo_config(algo_config_cls, normalize_advantage, exclude_dims):
    return algo_config_cls(
        share_param_critic=False,
        clip_epsilon=0.2,
        entropy_coef=0.0,
        critic_coef=1.0,
        loss_critic_type="l2",
        lmbda=0.9,
        scale_mapping="biased_softplus_1.0",
        use_tanh_normal=True,
        minibatch_advantage=False,
        normalize_advantage=normalize_advantage,
        normalize_advantage_exclude_dims=exclude_dims,
    )

@pytest.fixture(params=[(Ippo, IppoConfig), (Mappo, MappoConfig)], ids=["ippo", "mappo"])
def ppo_setup(request):
    algo_cls, algo_config_cls = request.param

    def _make(normalize_advantage, exclude_dims):
        algo_config = _make_ppo_config(algo_config_cls, normalize_advantage, exclude_dims)
        experiment = _make_experiment(algo_config)
        algo = algo_cls(**asdict(algo_config), experiment=experiment)
        continuous = not isinstance(experiment.action_spec["agents", "action"], (Categorical, OneHot))
        return algo, continuous
    return _make


@pytest.mark.parametrize("normalize_advantage,exclude_dims",[(True, [-2]), (True, []), (False, [-2])])
def test_ppo_advantage_normalization_init(ppo_setup, normalize_advantage, exclude_dims):
    algo, continuous = ppo_setup(normalize_advantage, exclude_dims)
    loss_module, use_target = algo._get_loss(
        "agents",
        policy_for_loss=TensorDictModule(lambda td, **kwargs: td, in_keys=[], out_keys=[]),
        continuous=continuous,
    )
    assert use_target is False
    assert isinstance(loss_module, ClipPPOLoss)
    assert loss_module.normalize_advantage is normalize_advantage
    assert loss_module.normalize_advantage_exclude_dims == exclude_dims


def test_ppo_advantage_normalization(ppo_setup):
    algo, continuous = ppo_setup(True, (-2,))
    policy = algo.get_policy_for_loss("agents")
    loss_module, _ = algo._get_loss("agents", policy, continuous=continuous)

    torch.manual_seed(0)
    batch, n_agents = 64, 2
    tensordict = TensorDict(batch_size=[batch])
    tensordict["agents", "observation"] = torch.rand(batch, n_agents, 3)
    tensordict["agents", "action"] = torch.randint(0, 2, (batch, n_agents))
    tensordict["agents", "log_prob"] = torch.zeros(batch, n_agents)
    tensordict["agents", "value_target"] = torch.zeros(batch, n_agents, 1)
    tensordict["agents", "advantage"] = torch.cat([torch.rand(batch, 1, 1), 5.0 * torch.rand(batch, 1, 1)], dim=1)

    def loss_with(exclude_dims):
        loss_module.normalize_advantage_exclude_dims = exclude_dims
        with warnings.catch_warnings():
            return loss_module(tensordict.clone())["loss_objective"].item()

    with torch.no_grad():
        joint = loss_with([])
        per_agent = loss_with((-2,))

    assert per_agent != pytest.approx(joint, rel=1e-3)
