import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from benchmarl.algorithms import algorithm_config_registry
from benchmarl.environments import task_config_registry
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.common import ModelConfig
from benchmarl.models.mlp import MlpConfig


def load_experiment_from_hydra_config(
    cfg: DictConfig, algo_name: str, task_name: str, model_config: ModelConfig
) -> Experiment:
    algorithm_config = algorithm_config_registry[algo_name](**cfg.algorithm)
    task_config = task_config_registry[task_name].update_config(cfg.task)
    experiment_config = ExperimentConfig(**cfg.experiment)

    return Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=cfg.seed,
        config=experiment_config,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    algo_name = hydra_choices.algorithm
    task_name = hydra_choices.task
    experiment = load_experiment_from_hydra_config(
        cfg,
        algo_name=algo_name,
        task_name=task_name,
        model_config=MlpConfig(
            num_cells=[64]
        ),  # Model still needs to be hydra configurable
    )
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
