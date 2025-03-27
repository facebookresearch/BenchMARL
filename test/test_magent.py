#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


import pytest

from benchmarl.algorithms import (
    algorithm_config_registry,
    IppoConfig,
    IsacConfig,
    MasacConfig,
    QmixConfig,
)
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import MAgentTask, Task
from benchmarl.experiment import Experiment

from utils import _has_magent2
from utils_experiment import ExperimentUtils


@pytest.mark.skipif(not _has_magent2, reason="magent2 not found")
class TestMagent:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("task", [MAgentTask.ADVERSARIAL_PURSUIT])
    def test_all_algos(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        cnn_sequence_config,
    ):

        # To not run unsupported algo-task pairs
        if not algo_config.supports_discrete_actions():
            pytest.skip()

        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=cnn_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", [IppoConfig, QmixConfig, IsacConfig])
    @pytest.mark.parametrize("task", [MAgentTask.ADVERSARIAL_PURSUIT])
    def test_gnn(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        cnn_gnn_sequence_config,
    ):
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=cnn_gnn_sequence_config,
            critic_model_config=cnn_gnn_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", [IppoConfig, QmixConfig, MasacConfig])
    @pytest.mark.parametrize("task", [MAgentTask.ADVERSARIAL_PURSUIT])
    def test_lstm(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        cnn_lstm_sequence_config,
    ):
        algo_config = algo_config.get_from_yaml()
        if algo_config.has_critic():
            algo_config.share_param_critic = False
        experiment_config.share_policy_params = False
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=cnn_lstm_sequence_config,
            critic_model_config=cnn_lstm_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", [IppoConfig])
    @pytest.mark.parametrize("task", [MAgentTask.ADVERSARIAL_PURSUIT])
    def test_reloading_trainer(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        cnn_sequence_config,
    ):
        # To not run unsupported algo-task pairs
        if not algo_config.supports_discrete_actions():
            pytest.skip()
        algo_config = algo_config.get_from_yaml()

        ExperimentUtils.check_experiment_loading(
            algo_config=algo_config,
            model_config=cnn_sequence_config,
            experiment_config=experiment_config,
            task=task.get_from_yaml(),
        )

    @pytest.mark.parametrize("algo_config", [QmixConfig, MasacConfig])
    @pytest.mark.parametrize("task", [MAgentTask.ADVERSARIAL_PURSUIT])
    @pytest.mark.parametrize("share_params", [True, False])
    def test_share_policy_params(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        share_params,
        experiment_config,
        cnn_sequence_config,
    ):
        experiment_config.share_policy_params = share_params
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=cnn_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()
