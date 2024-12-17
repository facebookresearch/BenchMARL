#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


import pytest

from benchmarl.algorithms import (
    algorithm_config_registry,
    IppoConfig,
    MappoConfig,
    MasacConfig,
    QmixConfig,
)
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import MeltingPotTask, Task
from benchmarl.experiment import Experiment

from utils import _has_meltingpot
from utils_experiment import ExperimentUtils


def _get_unique_envs(names):
    prefixes = set()
    result = []
    for env in names:
        prefix = env.name.split("_")[0]
        if prefix not in prefixes:
            prefixes.add(prefix)
            result.append(env)
    return result


@pytest.mark.skipif(not _has_meltingpot, reason="Meltingpot not found")
class TestMeltingPot:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("task", [MeltingPotTask.COMMONS_HARVEST__OPEN])
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
        experiment_config.checkpoint_interval = 0
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=cnn_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", [MasacConfig])
    @pytest.mark.parametrize("task", _get_unique_envs(list(MeltingPotTask))[:10])
    def test_all_tasks(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        cnn_sequence_config,
    ):
        task = task.get_from_yaml()
        experiment_config.checkpoint_interval = 0
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=cnn_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", [MappoConfig])
    @pytest.mark.parametrize("task", [MeltingPotTask.COINS])
    @pytest.mark.parametrize("parallel_collection", [True, False])
    def test_lstm(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        parallel_collection: bool,
        experiment_config,
        cnn_lstm_sequence_config,
    ):
        algo_config = algo_config.get_from_yaml()
        if algo_config.has_critic():
            algo_config.share_param_critic = False
        experiment_config.parallel_collection = parallel_collection
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

    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("task", [MeltingPotTask.COMMONS_HARVEST__OPEN])
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

    @pytest.mark.parametrize("algo_config", [QmixConfig, IppoConfig, MasacConfig])
    @pytest.mark.parametrize("task", [MeltingPotTask.COMMONS_HARVEST__OPEN])
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
        experiment_config.checkpoint_interval = 0
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=cnn_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()
