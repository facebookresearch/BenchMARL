import hydra

from benchmarl.environments import task_config_registry
from benchmarl.experiment import Experiment

from benchmarl.hydra_run import load_model_from_hydra_config
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../benchmarl/conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task

    algorithm_config = OmegaConf.to_object(cfg.algorithm)
    experiment_config = OmegaConf.to_object(cfg.experiment)
    task_config = task_config_registry[task_name].update_config(cfg.task)
    model_config = load_model_from_hydra_config(cfg.model)

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
