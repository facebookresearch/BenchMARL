import hydra

from benchmarl.experiment import load_experiment_from_hydra_config
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../benchmarl/conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    algo_name = hydra_choices.algorithm
    task_name = hydra_choices.task
    experiment = load_experiment_from_hydra_config(
        cfg, algo_name=algo_name, task_name=task_name
    )
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
