import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from benchmarl.environments import task_config_registry
from benchmarl.experiment import Experiment
from benchmarl.models import model_config_registry
from benchmarl.models.common import ModelConfig, parse_model_config, SequenceModelConfig


def load_experiment_from_hydra_config(cfg: DictConfig, task_name: str) -> Experiment:
    algorithm_config = OmegaConf.to_object(cfg.algorithm)
    experiment_config = OmegaConf.to_object(cfg.experiment)
    task_config = task_config_registry[task_name].update_config(cfg.task)
    model_config = load_model_from_hydra_config(cfg.model)

    return Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=cfg.seed,
        config=experiment_config,
    )


def load_model_from_hydra_config(cfg: DictConfig) -> ModelConfig:
    if "layers" in cfg.keys():
        model_configs = [
            load_model_from_hydra_config(cfg.layers[f"l{i}"])
            for i in range(1, len(cfg.layers) + 1)
        ]
        return SequenceModelConfig(
            model_configs=model_configs, intermediate_sizes=cfg.intermediate_sizes
        )
    else:
        model_class = model_config_registry[cfg.name]
        return model_class(**parse_model_config(OmegaConf.to_container(cfg)))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    experiment = load_experiment_from_hydra_config(
        cfg,
        task_name=task_name,
    )
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
