#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from omegaconf import DictConfig, OmegaConf

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task, task_config_registry
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import model_config_registry
from benchmarl.models.common import ModelConfig, parse_model_config, SequenceModelConfig


def load_experiment_from_hydra(cfg: DictConfig, task_name: str) -> Experiment:
    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    task_config = load_task_config_from_hydra(cfg.task, task_name)
    model_config = load_model_config_from_hydra(cfg.model)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)

    return Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
    )


def load_task_config_from_hydra(cfg: DictConfig, task_name: str) -> Task:
    return task_config_registry[task_name].update_config(
        OmegaConf.to_container(cfg, resolve=True)
    )


def load_experiment_config_from_hydra(cfg: DictConfig) -> ExperimentConfig:
    return OmegaConf.to_object(cfg)


def load_algorithm_config_from_hydra(cfg: DictConfig) -> AlgorithmConfig:
    return OmegaConf.to_object(cfg)


def load_model_config_from_hydra(cfg: DictConfig) -> ModelConfig:
    if "layers" in cfg.keys():
        model_configs = [
            load_model_config_from_hydra(cfg.layers[f"l{i}"])
            for i in range(1, len(cfg.layers) + 1)
        ]
        return SequenceModelConfig(
            model_configs=model_configs, intermediate_sizes=cfg.intermediate_sizes
        )
    else:
        model_class = model_config_registry[cfg.name]
        return model_class(
            **parse_model_config(OmegaConf.to_container(cfg, resolve=True))
        )
