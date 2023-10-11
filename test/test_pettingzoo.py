#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import importlib

import pytest

from benchmarl.algorithms import (
    algorithm_config_registry,
    IppoConfig,
    MaddpgConfig,
    MasacConfig,
    QmixConfig,
    VdnConfig,
)
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import PettingZooTask, Task
from benchmarl.experiment import Experiment
from utils_experiment import ExperimentUtils

_has_pettingzoo = importlib.util.find_spec("pettingzoo") is not None


@pytest.mark.skipif(not _has_pettingzoo, reason="PettingZoo not found")
class TestPettingzoo:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("prefer_continuous", [True, False])
    @pytest.mark.parametrize(
        "task", [PettingZooTask.MULTIWALKER, PettingZooTask.SIMPLE_TAG]
    )
    def test_all_algos(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        prefer_continuous,
        experiment_config,
        mlp_sequence_config,
    ):
        # To not run the same test twice
        if (prefer_continuous and not algo_config.supports_continuous_actions()) or (
            not prefer_continuous and not algo_config.supports_discrete_actions()
        ):
            return

        # To not run unsupported algo-task pairs
        if (
            not task.supports_continuous_actions()
            and not algo_config.supports_discrete_actions()
        ) or (
            not task.supports_discrete_actions()
            and not algo_config.supports_continuous_actions()
        ):
            return

        task = task.get_from_yaml()
        experiment_config.prefer_continuous_actions = prefer_continuous
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", [IppoConfig, MasacConfig])
    @pytest.mark.parametrize("task", list(PettingZooTask))
    def test_all_tasks(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        mlp_sequence_config,
    ):

        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("prefer_continuous", [True, False])
    @pytest.mark.parametrize("task", [PettingZooTask.SIMPLE_TAG])
    def test_reloading_trainer(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        mlp_sequence_config,
        prefer_continuous,
    ):
        experiment_config.prefer_continuous_actions = prefer_continuous
        algo_config = algo_config.get_from_yaml()
        if isinstance(algo_config, VdnConfig):
            # There are some bugs currently in TorchRL
            return
        ExperimentUtils.check_experiment_loading(
            algo_config=algo_config,
            model_config=mlp_sequence_config,
            experiment_config=experiment_config,
            task=task.get_from_yaml(),
        )

    @pytest.mark.parametrize(
        "algo_config", [QmixConfig, IppoConfig, MaddpgConfig, MasacConfig]
    )
    @pytest.mark.parametrize("task", [PettingZooTask.SIMPLE_TAG])
    @pytest.mark.parametrize("share_params", [True, False])
    def test_share_policy_params(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        share_params,
        experiment_config,
        mlp_sequence_config,
    ):
        experiment_config.share_policy_params = share_params
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()
