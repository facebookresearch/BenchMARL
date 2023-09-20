import importlib

import pytest

from benchmarl.algorithms import algorithm_config_registry
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task, VmasTask
from benchmarl.experiment import Experiment

_has_vmas = importlib.util.find_spec("vmas") is not None


@pytest.mark.skipif(not _has_vmas, reason="VMAS not found")
class TestVmas:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("continuous", [True])
    @pytest.mark.parametrize("task", list(VmasTask))
    def test_all_algos_all_tasks(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        continuous,
        experiment_config,
        mlp_sequence_config,
    ):
        task = task.get_from_yaml()

        experiment_config.prefer_continuous_actions = continuous

        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()
