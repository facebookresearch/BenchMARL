import os
from pathlib import Path

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    experiment_config = ExperimentConfig.get_from_yaml()
    # Save the experiment in the current folder
    experiment_config.save_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    # Checkpoint at every iteration
    experiment_config.checkpoint_interval = 1
    # Run 3 iterations
    experiment_config.n_iters = 3

    task = VmasTask.BALANCE.get_from_yaml()
    algorithm_config = MappoConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment.run()

    # Now we tell it where to restore  from
    experiment_config.restore_file = (
        experiment.folder_name
        / "checkpoints"
        / f"checkpoint_{experiment_config.n_iters}.pt"
    )
    # The experiment will be saved in the ame folder as the one it is restoring from
    experiment_config.save_folder = None
    # Let's do 3 more iters
    experiment_config.n_iters += 3

    experiment = Experiment(
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=0,
        config=experiment_config,
        task=task,
    )
    experiment.run()
