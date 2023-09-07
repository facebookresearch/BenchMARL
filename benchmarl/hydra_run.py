from dataclasses import dataclass, MISSING

import hydra

from algorithms import algorithm_config_registry, MappoConfig
from algorithms.common import AlgorithmConfig
from environments import task_config_registry
from experiment import Experiment, ExperimentConfig
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

cs = ConfigStore.instance()
cs.store(name="experiment_config", group="experiment", node=ExperimentConfig)
for algo_name, algo_config in algorithm_config_registry.items():
    cs.store(name=f"{algo_name}_config", group="algorithm", node=algo_config)
for task_name, (task_config, _) in task_config_registry.items():
    cs.store(
        name=f"{task_name.replace('/','_')}_config", group="task", node=task_config
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices

    algorithm_config = algorithm_config_registry[hydra_choices.algorithm](
        **cfg.algorithm
    )
    task_config = task_config_registry[hydra_choices.task][1].update_config(cfg.task)
    experiment_config = ExperimentConfig(**cfg.experiment)

    MappoConfig.get_from_yaml()
    # experiment_config = ExperimentConfig(**cfg.experiment)
    # algorithm_config = algorithm_config_registry[cfg.algorithm.name](**cfg.algorithm)


if __name__ == "__main__":
    my_app()
