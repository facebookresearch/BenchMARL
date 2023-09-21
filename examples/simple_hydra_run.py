import hydra
from benchmarl.hydra_config import load_experiment_from_hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../benchmarl/conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    print(f"\nAlgorithm: {hydra_choices.algorithm}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    experiment = load_experiment_from_hydra(
        cfg,
        task_name=task_name,
    )
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()

    # You can run multiple experiments like so
    # python simple_hydra_run.py --multirun algorithm=mappo,qmix,maddpg,masac task=vmas/balance
