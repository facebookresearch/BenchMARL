#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import importlib
import os
import os.path as osp
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tensordict import TensorDictBase
from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase

from benchmarl.utils import DEVICE_TYPING, read_yaml_config


def _load_config(name: str, config: Dict[str, Any]):
    if not name.endswith(".py"):
        name += ".py"

    pathname = None
    for dirpath, _, filenames in os.walk(osp.dirname(__file__)):
        if pathname is None:
            for filename in filenames:
                if filename == name:
                    pathname = os.path.join(dirpath, filename)
                    break

    if pathname is None:
        raise ValueError(f"Task {name} not found.")

    spec = importlib.util.spec_from_file_location("", pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TaskConfig(**config).__dict__


class Task(Enum):
    """Task.

    Tasks are enums, one enum for each environment.
    Each enum member has a config attribute that is a dictionary which can be loaded from .yaml
    files. You can also access and modify this attribute directly.

    Each new environment should inherit from Task and instantiate its members as

    TASK_1 = None
    TASK_2 = None
    ...

    Tasks configs are loaded from benchmarl/conf/environments
    """

    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def update_config(self, config: Dict[str, Any]) -> Task:
        """
        Updates the task config

        Args:
            config (dictionary): The config to update in the task

        Returns: The updated task

        """
        if self.config is None:
            self.config = config
        else:
            self.config.update(config)
        return self

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        """
        This function is used to obtain a TorchRL object from the enum Task.

        Args:
            num_envs (int): The number of envs that should be in the batch_size of the returned env.
                In vectorized envs, this can be used to set the number of batched environments.
                If your environment is not vectorized, you can just ignore this, and it will be
                wrapped in a torchrl.envs.SerialEnv with num_envs automatically.
            continuous_actions (bool): Whether your environment should have continuous or discrete actions.
                If your environment does not support both, ignore this and refer to the supports_x_actions methods.
            seed (optional, int): The seed of your env
            device (str): the device of your env, you can pass this to any torchrl env constructor

        Returns: a function that takes no arguments and returns a torchrl.envs.EnvBase object

        """
        raise NotImplementedError

    def supports_continuous_actions(self) -> bool:
        """
        Return true if your task supports continuous actions.
        If true, self.get_env_fun might be called with continuous_actions=True
        """
        raise NotImplementedError

    def supports_discrete_actions(self) -> bool:
        """
        Return true if your task supports discrete actions.
        If true, self.get_env_fun might be called with continuous_actions=False
        """
        raise NotImplementedError

    def max_steps(self, env: EnvBase) -> int:
        """
        The maximum number of steps allowed in an evaluation rollout.

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    def has_render(self, env: EnvBase) -> bool:
        """
        If env.render() should be called on the environment

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        """
        The group_map mapping agents groups to agent names.
        This should be reelected in the TensorDicts coming from the environment where
        agent data is supposed to be stacked according to this.

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        """
        A spec for the observation.
        Must be a CompositeSpec with one (group_name, "observation") entry per group.

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        """
        A spec for the info.
        If provided, must be a CompositeSpec with one (group_name, "info") entry per group (this entry can be composite).


        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        """
        A spec for the state.
        If provided, must be a CompositeSpec with one "state" entry.

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        """
        A spec for the action.
        If provided, must be a CompositeSpec with one (group_name, "action") entry per group.

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        """
        A spec for the action mask.
        If provided, must be a CompositeSpec with one (group_name, "action_mask") entry per group.

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    @staticmethod
    def env_name() -> str:
        """
        The name of the environment in the benchmarl/conf/task folder

        """
        raise NotImplementedError

    @staticmethod
    def log_info(batch: TensorDictBase) -> Dict[str, float]:
        """
        Return a str->float dict with extra items to log.
        This function has access to the collected batch and is optional.

        Args:
            batch (TensorDictBase): the batch obtained from collection.

        """
        return {}

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}: (config={self.config})"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def _load_from_yaml(name: str) -> Dict[str, Any]:
        yaml_path = Path(__file__).parent.parent / "conf" / "task" / f"{name}.yaml"
        return read_yaml_config(str(yaml_path.resolve()))

    def get_from_yaml(self, path: Optional[str] = None) -> Task:
        """
        Load the task configuration from yaml

        Args:
            path (str, optional): The full path of the yaml file to load from. If None, it will default to
                benchmarl/conf/task/self.env_name()/self.name

        Returns: the task with the loaded config
        """
        if path is None:
            task_name = self.name.lower()
            return self.update_config(
                Task._load_from_yaml(str(Path(self.env_name()) / Path(task_name)))
            )
        else:
            return self.update_config(**read_yaml_config(path))
