import hydra
from benchmarl.environments import task_config_registry
from benchmarl.experiment import Experiment, ExperimentConfig

from benchmarl.hydra_run import load_model_from_hydra_config
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../benchmarl/conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    print(f"\nAlgorithm: {hydra_choices.algorithm}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    algorithm_config = OmegaConf.to_object(cfg.algorithm)
    task_config = task_config_registry[task_name].update_config(
        OmegaConf.to_container(cfg.task, resolve=True)
    )
    model_config = load_model_from_hydra_config(cfg.model)
    experiment_config: ExperimentConfig = OmegaConf.to_object(cfg.experiment)

    # Hyperparameter changes for VMAS experiments
    experiment_config.sampling_device = "cuda"
    experiment_config.train_device = "cuda"
    experiment_config.collected_frames_per_batch = 60_000
    experiment_config.n_envs_per_worker = 600
    experiment_config.on_policy_minibatch_size = 4096
    experiment_config.evaluation_episodes = 200

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
    # To reproduce the VMAS results launch this with
    # python run.py algorithm=ippo "task=vmas/navigation,vmas/balance,vmas/sampling" "seed=0,1,2"
