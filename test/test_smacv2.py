#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import importlib

import pytest

from benchmarl.algorithms import algorithm_config_registry, MappoConfig, QmixConfig
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Smacv2Task
from benchmarl.experiment import Experiment

_has_smacv2 = importlib.util.find_spec("smacv2") is not None


@pytest.mark.skipif(not _has_smacv2, reason="SMACv2 not found")
class TestSmacv2:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("task", [Smacv2Task.PROTOSS_5_VS_5])
    def test_all_algos(
        self, algo_config: AlgorithmConfig, task, experiment_config, mlp_sequence_config
    ):
        if algo_config.supports_discrete_actions():
            task = task.get_from_yaml()

            experiment = Experiment(
                algorithm_config=algo_config.get_from_yaml(),
                model_config=mlp_sequence_config,
                seed=0,
                config=experiment_config,
                task=task,
            )
            experiment.run()

    @pytest.mark.parametrize("algo_config", [QmixConfig, MappoConfig])
    @pytest.mark.parametrize(
        "task",
        [
            Smacv2Task.PROTOSS_5_VS_5,
            Smacv2Task.ZERG_5_VS_5,
            Smacv2Task.TERRAN_5_VS_5,
        ],
    )
    def test_all_tasks(
        self, algo_config: AlgorithmConfig, task, experiment_config, mlp_sequence_config
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
