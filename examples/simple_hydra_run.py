import hydra

from benchmarl.algorithms import algorithm_config_registry
from benchmarl.environments import task_config_registry
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.common import SequenceModelConfig
from benchmarl.models.mlp import MlpConfig
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../benchmarl/conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    algo_name = hydra_choices.algorithm
    task_name = hydra_choices.task

    algorithm_config = algorithm_config_registry[algo_name](**cfg.algorithm)
    task_config = task_config_registry[task_name].update_config(cfg.task)
    experiment_config = ExperimentConfig(**cfg.experiment)

    # Model still need to be refactored for hydra loading
    model_config = SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[64, 64]),
            MlpConfig(num_cells=[256]),
        ],
        intermediate_sizes=[128],
    )

    experiment = Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=cfg.seed,
        config=experiment_config,
    )

    experiment.run()


if __name__ == "__main__":
    hydra_experiment()

    # You can run multiple experiments like so
    # python simple_hydra_run.py --multirun algorithm=mappo,qmix,maddpg,masac task=vmas/balance
