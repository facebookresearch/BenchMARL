import importlib

import pytest

from benchmarl.algorithms import algorithm_config_registry, MappoConfig, QmixConfig
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task, VmasTask
from benchmarl.experiment import Experiment

_has_vmas = importlib.util.find_spec("vmas") is not None


@pytest.mark.skipif(not _has_vmas, reason="VMAS not found")
class TestVmas:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("continuous", [True, False])
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

    @pytest.mark.parametrize("algo_config", [MappoConfig, QmixConfig])
    @pytest.mark.parametrize("task", [VmasTask.BALANCE, VmasTask.SAMPLING])
    def test_reloading_trainer(
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
        policy = experiment.policy
        losses = experiment.losses
        exp_folder = experiment.folder_name
        experiment_config.n_iters = 6
        experiment_config.restore_file = exp_folder / "checkpoints" / "checkpoint_3.pt"
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        for param1, param2 in zip(
            list(experiment.policy.parameters()), list(policy.parameters())
        ):
            assert (param1 == param2).all()
        for loss1, loss2 in zip(experiment.losses.values(), losses.values()):
            for param1, param2 in zip(
                list(loss1.parameters()), list(loss2.parameters())
            ):
                assert (param1 == param2).all()
        assert experiment.n_iters_performed == 3
        assert experiment.folder_name == exp_folder
        experiment.run()
        assert experiment.n_iters_performed == 6
