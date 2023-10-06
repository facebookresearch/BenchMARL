from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.common import ModelConfig


class ExperimentUtils:
    @staticmethod
    def check_experiment_loading(
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config: ExperimentConfig,
        model_config: ModelConfig,
    ):
        max_n_iters = experiment_config.max_n_iters
        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

        policy = experiment.policy
        losses = experiment.losses
        exp_folder = experiment.folder_name

        experiment_config.max_n_iters = max_n_iters + 3
        experiment_config.restore_file = (
            exp_folder / "checkpoints" / f"checkpoint_{max_n_iters}.pt"
        )
        experiment_config.save_folder = None
        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=model_config,
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
        assert experiment.n_iters_performed == max_n_iters
        assert experiment.folder_name == exp_folder
        experiment.run()
        assert experiment.n_iters_performed == max_n_iters + 3
