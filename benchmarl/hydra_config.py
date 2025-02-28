#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import importlib
from dataclasses import is_dataclass
from pathlib import Path

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import task_config_registry, TaskClass
from benchmarl.environments.common import _type_check_task_config
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import model_config_registry
from benchmarl.models.common import ModelConfig, parse_model_config, SequenceModelConfig

_has_hydra = importlib.util.find_spec("hydra") is not None

if _has_hydra:
    from hydra import compose, initialize, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf


class _HydraMissingMetadataError(FileNotFoundError):
    def __init__(
        self,
        message=".hydra folder not found (should be max 3 levels above checkpoint file",
    ):
        self.message = message
        super().__init__(self.message)


def load_experiment_from_hydra(
    cfg: DictConfig, task_name: str, callbacks=()
) -> Experiment:
    """Creates an :class:`~benchmarl.experiment.Experiment` from hydra config.

    Args:
        cfg (DictConfig): the config dictionary from hydra main
        task_name (str): the name of the task to load

    Returns:
        :class:`~benchmarl.experiment.Experiment`

    """
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
        callbacks=callbacks,
    )


def load_task_config_from_hydra(cfg: DictConfig, task_name: str) -> TaskClass:
    """Returns a :class:`~benchmarl.environments.Task` from hydra config.

    Args:
        cfg (DictConfig): the task config dictionary from hydra
        task_name (str): the name of the task to load

    Returns:
        :class:`~benchmarl.environments.Task`

    """
    environment_name, inner_task_name = task_name.split("/")
    cfg_dict_checked = OmegaConf.to_object(cfg)
    if is_dataclass(cfg_dict_checked):
        cfg_dict_checked = cfg_dict_checked.__dict__
    cfg_dict_checked = _type_check_task_config(
        environment_name, inner_task_name, cfg_dict_checked
    )  # Only needed for the warning
    return task_config_registry[task_name].get_task(cfg_dict_checked)


def load_experiment_config_from_hydra(cfg: DictConfig) -> ExperimentConfig:
    """Returns a :class:`~benchmarl.experiment.ExperimentConfig` from hydra config.

    Args:
        cfg (DictConfig): the experiment config dictionary from hydra

    Returns:
        :class:`~benchmarl.experiment.ExperimentConfig`

    """
    return OmegaConf.to_object(cfg)


def load_algorithm_config_from_hydra(cfg: DictConfig) -> AlgorithmConfig:
    """Returns a :class:`~benchmarl.algorithms.AlgorithmConfig` from hydra config.

    Args:
        cfg (DictConfig): the algorithm config dictionary from hydra

    Returns:
        :class:`~benchmarl.algorithms.AlgorithmConfig`

    """
    return OmegaConf.to_object(cfg)


def load_model_config_from_hydra(cfg: DictConfig) -> ModelConfig:
    """Returns a :class:`~benchmarl.models.ModelConfig` from hydra config.

    Args:
        cfg (DictConfig): the model config dictionary from hydra

    Returns:
        :class:`~benchmarl.models.ModelConfig`

    """
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
            **parse_model_config(
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            )
        )


def _find_hydra_folder(restore_file: str) -> str:
    """Given the restore file, look for the .hydra folder max three levels above it."""
    current_folder = Path(restore_file).parent.resolve()
    for _ in range(3):
        hydra_dir = current_folder / ".hydra"
        if hydra_dir.exists() and hydra_dir.is_dir():
            return str(hydra_dir)
        current_folder = current_folder.parent
    raise _HydraMissingMetadataError()


def reload_experiment_from_file(restore_file: str) -> Experiment:
    """Reloads the experiment from a given restore file.

    Requires a ``.hydra`` folder containing ``config.yaml``, ``hydra.yaml``, and ``overrides.yaml``
    at max three directory levels higher than the checkpoint file. This should be automatically created by hydra.

    Args:
        restore_file (str): The checkpoint file of the experiment reload.

    """
    try:
        hydra_folder = _find_hydra_folder(restore_file)
    except _HydraMissingMetadataError:
        # Hydra was not used
        return Experiment.reload_from_file(restore_file)

    with initialize(
        version_base=None,
        config_path="conf",
    ):
        cfg = compose(
            config_name="config",
            overrides=OmegaConf.load(Path(hydra_folder) / "overrides.yaml"),
            return_hydra_config=True,
        )
        task_name = cfg.hydra.runtime.choices.task
        algorithm_name = cfg.hydra.runtime.choices.algorithm
    with initialize_config_dir(version_base=None, config_dir=hydra_folder):
        cfg_loaded = dict(compose(config_name="config"))

    for key in ("experiment", "algorithm", "task", "model", "critic_model"):
        cfg[key].update(cfg_loaded[key])
        cfg_loaded.pop(key)

    cfg.update(cfg_loaded)
    del cfg.hydra
    cfg.experiment.restore_file = restore_file

    print("\nReloaded experiment with:")
    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    return load_experiment_from_hydra(cfg, task_name=task_name)
