import importlib

import pytest

from benchmarl.algorithms import (
    algorithm_config_registry,
    MappoConfig,
    MasacConfig,
    QmixConfig,
)
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Smacv2Task
from benchmarl.experiment import Experiment


_has_smacv2 = importlib.util.find_spec("smacv2") is not None


@pytest.mark.skipif(not _has_smacv2, reason="SMACv2 not found")
class TestSmacv2:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("task", list(Smacv2Task))
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

    @pytest.mark.parametrize("algo_config", [QmixConfig, MappoConfig, MasacConfig])
    @pytest.mark.parametrize("task", list(Smacv2Task))
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
