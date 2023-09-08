import hydra

from benchmarl.algorithms import algorithm_config_registry
from benchmarl.environments import task_config_registry
from benchmarl.experiment import Experiment, ExperimentConfig

from benchmarl.models.mlp import MlpConfig
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../benchmarl/conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices

    algorithm_config = algorithm_config_registry[hydra_choices.algorithm](
        **cfg.algorithm
    )
    task_config = task_config_registry[hydra_choices.task].update_config(cfg.task)
    experiment_config = ExperimentConfig(**cfg.experiment)

    model_config = MlpConfig(num_cells=[64, 64])

    experiment = Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=cfg.seed,
        config=experiment_config,
    )
    experiment.run()


if __name__ == "__main__":
    my_app()
