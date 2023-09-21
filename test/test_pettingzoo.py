import importlib

import pytest

from benchmarl.algorithms import (
    algorithm_config_registry,
    IppoConfig,
    MaddpgConfig,
    MappoConfig,
    MasacConfig,
    QmixConfig,
)
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import PettingZooTask, Task
from benchmarl.experiment import Experiment
from utils_experiment import ExperimentUtils

_has_pettingzoo = importlib.util.find_spec("pettingzoo") is not None


@pytest.mark.skipif(not _has_pettingzoo, reason="PettingZoo not found")
class TestPettingzoo:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("continuous", [True, False])
    @pytest.mark.parametrize("task", list(PettingZooTask))
    def test_all_algos_all_tasks(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        continuous,
        experiment_config,
        mlp_sequence_config,
    ):
        if (
            not task.supports_continuous_actions()
            and (continuous or not algo_config.supports_discrete_actions())
        ) or (
            not task.supports_discrete_actions()
            and (not continuous or not algo_config.supports_continuous_actions())
        ):
            return

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

    @pytest.mark.parametrize("algo_config", [MappoConfig, QmixConfig])
    @pytest.mark.parametrize(
        "task", [PettingZooTask.SIMPLE_TAG, PettingZooTask.SIMPLE_TAG]
    )
    def test_reloading_trainer(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        mlp_sequence_config,
    ):
        ExperimentUtils.check_experiment_loading(
            algo_config=algo_config.get_from_yaml(),
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
