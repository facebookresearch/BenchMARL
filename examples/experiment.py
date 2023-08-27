from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    task = VmasTask.BALANCE
    model_config = MlpConfig()
    algorithm_config = AlgorithmConfig()

    experiment_config = ExperimentConfig(
        task=task, algorithm_config=algorithm_config, model_config=model_config
    )
    experiment = experiment_config.get_experiment()

    experiment.run()
