#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


import pytest

from benchmarl.algorithms import (
    algorithm_config_registry,
    IddpgConfig,
    IppoConfig,
    IqlConfig,
    IsacConfig,
    MaddpgConfig,
    MappoConfig,
    MasacConfig,
    QmixConfig,
)
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import PettingZooTask, Task
from benchmarl.experiment import Experiment

from utils import _has_pettingzoo
from utils_experiment import ExperimentUtils


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
            pytest.skip()

        task = task.get_from_yaml()
        # To not run unsupported algo-task pairs
        if (
            not task.supports_continuous_actions()
            and not algo_config.supports_discrete_actions()
        ) or (
            not task.supports_discrete_actions()
            and not algo_config.supports_continuous_actions()
        ):
            pytest.skip()

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

    @pytest.mark.parametrize("algo_config", [IppoConfig, QmixConfig, IsacConfig])
    @pytest.mark.parametrize("task", [PettingZooTask.SIMPLE_TAG])
    def test_gnn(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        mlp_gnn_sequence_config,
    ):
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_gnn_sequence_config,
            critic_model_config=mlp_gnn_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize(
        "algo_config", [IddpgConfig, MappoConfig, QmixConfig, MasacConfig]
    )
    @pytest.mark.parametrize("task", [PettingZooTask.SIMPLE_TAG])
    @pytest.mark.parametrize("parallel_collection", [True, False])
    def test_gru(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        parallel_collection: bool,
        experiment_config,
        gru_mlp_sequence_config,
    ):
        algo_config = algo_config.get_from_yaml()
        if algo_config.has_critic():
            algo_config.share_param_critic = False
        experiment_config.parallel_collection = parallel_collection
        experiment_config.share_policy_params = False
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=gru_mlp_sequence_config,
            critic_model_config=gru_mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize(
        "algo_config", [MaddpgConfig, IppoConfig, QmixConfig, IsacConfig]
    )
    @pytest.mark.parametrize("task", [PettingZooTask.SIMPLE_TAG])
    def test_lstm(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        lstm_mlp_sequence_config,
    ):
        algo_config = algo_config.get_from_yaml()
        if algo_config.has_critic():
            algo_config.share_param_critic = False
        experiment_config.share_policy_params = False
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=lstm_mlp_sequence_config,
            critic_model_config=lstm_mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", [MappoConfig, IsacConfig, IqlConfig])
    @pytest.mark.parametrize("prefer_continuous", [True, False])
    @pytest.mark.parametrize("task", [PettingZooTask.SIMPLE_TAG])
    @pytest.mark.parametrize("parallel_collection", [True, False])
    def test_reloading_trainer(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        parallel_collection,
        experiment_config,
        mlp_sequence_config,
        prefer_continuous,
    ):
        # To not run the same test twice
        if (prefer_continuous and not algo_config.supports_continuous_actions()) or (
            not prefer_continuous and not algo_config.supports_discrete_actions()
        ):
            pytest.skip()

        experiment_config.parallel_collection = parallel_collection
        experiment_config.prefer_continuous_actions = prefer_continuous
        algo_config = algo_config.get_from_yaml()

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
