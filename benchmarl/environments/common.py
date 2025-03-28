#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import abc

import importlib

import warnings
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from tensordict import TensorDictBase
from torch import Tensor

from torchrl.data import Composite
from torchrl.envs import EnvBase, RewardSum, Transform

from benchmarl.utils import _read_yaml_config, DEVICE_TYPING


def _type_check_task_config(
    environemnt_name: str,
    task_name: str,
    config: Dict[str, Any],
    warn_on_missing_dataclass: bool = True,
):

    task_config_class = _get_task_config_class(environemnt_name, task_name)

    if task_config_class is not None:
        return task_config_class(**config).__dict__
    else:
        if warn_on_missing_dataclass:
            warnings.warn(
                "TaskConfig python dataclass not found, task is being loaded without type checks"
            )
        return config


def _get_task_config_class(environemnt_name: str, task_name: str):
    try:
        module = importlib.import_module(
            f"{'.'.join(__name__.split('.')[:-1])}.{environemnt_name}.{task_name}"
        )
        return module.TaskConfig
    except ModuleNotFoundError:
        return None


class TaskClass(abc.ABC):
    """
    The class associated with an environment.

    This class contains the logic on how to construct tasks for a specific environment.

    The :meth:`TaskClass.get_env_fun` is the core method of this class, which will define the logic on how to construct
    a :class:`torchrl.EnvBase` given :meth:`TaskClass.config` and :meth:`TaskClass.name`.

    This methods, algonside all other abstract ones, is what users need to implement to introiduce new tasks.

    Args:
       name (str): The name of the task
       config (Dict[str, Any]): The configuration of the task

    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        if config is None:
            config = {}
        self.config = config

    @abstractmethod
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
                wrapped in a :class:`torchrl.envs.SerialEnv` with num_envs automatically.
            continuous_actions (bool): Whether your environment should have continuous or discrete actions.
                If your environment does not support both, ignore this and refer to the supports_x_actions methods.
            seed (optional, int): The seed of your env
            device (str): the device of your env, you can pass this to any torchrl env constructor

        Returns: a function that takes no arguments and returns a :class:`torchrl.envs.EnvBase` object

        """
        raise NotImplementedError

    @abstractmethod
    def supports_continuous_actions(self) -> bool:
        """
        Return true if your task supports continuous actions.
        If true, self.get_env_fun might be called with continuous_actions=True
        """
        raise NotImplementedError

    @abstractmethod
    def supports_discrete_actions(self) -> bool:
        """
        Return true if your task supports discrete actions.
        If true, self.get_env_fun might be called with continuous_actions=False
        """
        raise NotImplementedError

    @abstractmethod
    def max_steps(self, env: EnvBase) -> int:
        """
        The maximum number of steps allowed in an evaluation rollout.

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    @abstractmethod
    def has_render(self, env: EnvBase) -> bool:
        """
        If env.render() should be called on the environment

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    @abstractmethod
    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        """
        The group_map mapping agents groups to agent names.
        This should be reelected in the TensorDicts coming from the environment where
        agent data is supposed to be stacked according to this.

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    @abstractmethod
    def observation_spec(self, env: EnvBase) -> Composite:
        """
        A spec for the observation.
        Must be a Composite with as many entries as needed nested under the ``group_name`` key.

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        Examples:
            >>> print(task.observation_spec(env))
            Composite(
                agents: Composite(
                    observation: Composite(
                        image: UnboundedDiscreteTensorSpec(
                            shape=torch.Size([8, 88, 88, 3]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([8, 88, 88, 3]), device=cpu, dtype=torch.int64, contiguous=True),
                                high=Tensor(shape=torch.Size([8, 88, 88, 3]), device=cpu, dtype=torch.int64, contiguous=True)),
                            device=cpu,
                            dtype=torch.uint8,
                            domain=discrete),
                        array: Unbounded(
                            shape=torch.Size([8, 3]),
                            space=None,
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous), device=cpu, shape=torch.Size([8])), device=cpu, shape=torch.Size([8])), device=cpu, shape=torch.Size([]))


        """
        raise NotImplementedError

    @abstractmethod
    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        """
        A spec for the info.
        If provided, must be a Composite with one (group_name, "info") entry per group (this entry can be composite).


        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    @abstractmethod
    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        """
        A spec for the state.
        If provided, must be a Composite with one entry.

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    @abstractmethod
    def action_spec(self, env: EnvBase) -> Composite:
        """
        A spec for the action.
        If provided, must be a Composite with one (group_name, "action") entry per group.

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    @abstractmethod
    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        """
        A spec for the action mask.
        If provided, must be a Composite with one (group_name, "action_mask") entry per group.

        Args:
            env (EnvBase): An environment created via self.get_env_fun

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def env_name() -> str:
        """
        The name of the environment in the ``benchmarl/conf/task`` folder
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

    def get_reward_sum_transform(self, env: EnvBase) -> Transform:
        """
        Returns the RewardSum transform for the environment

        Args:
            env (EnvBase): An environment created via self.get_env_fun
        """
        if "_reset" in env.reset_keys:
            reset_keys = ["_reset"] * len(self.group_map(env).keys())
        else:
            reset_keys = env.reset_keys
        return RewardSum(reset_keys=reset_keys)

    def get_env_transforms(self, env: EnvBase) -> List[Transform]:
        """
        Returns a list of :class:`torchrl.envs.Transform` to be applied to the env.

        Args:
            env (EnvBase): An environment created via self.get_env_fun


        """
        return []

    def get_replay_buffer_transforms(self, env: EnvBase, group: str) -> List[Transform]:
        """
        Returns a list of :class:`torchrl.envs.Transform` to be applied to the :class:`torchrl.data.ReplayBuffer`
        of the specified group.

        Args:
            env (EnvBase): An environment created via self.get_env_fun
            group (str): The agent group using the replay buffer

        """
        return []

    @staticmethod
    def render_callback(experiment, env: EnvBase, data: TensorDictBase) -> Tensor:
        """
        The render callback function for the enviornment.

        This function is called at every step during evaluation to provide pixels for rendering.

        Args:
            experiment (Experiment): The Benchmarl experiment.
            env (EnvBase): An environment created via self.get_env_fun()()
            data (TensorDictBase): the current rollout data from the enviornment.

        Returns:
            The :class:`torch.Tensor` containing the pixels


        """
        try:
            return env.render(mode="rgb_array")
        except TypeError:
            return env.render()

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}: (config={self.config})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.config == other.config


class Task(Enum):
    """Enum of tasks in an environment.

    Tasks are enums, one enum for each environment.
    Each enviornment usually contains multiple tasks.

    Tasks are used just to enumerate the available tasks, to convert a :class:`Task` into its corresponding instantiation,
    you can call :meth:`Task.get_from_yaml` which will load the task config form yaml into the associated :class:`TaskClass`.

    Each enum member can also be converted to a :class:`TaskClass` by calling :meth:`Task.get_task`, (which by default behaves like
    :meth:`Task.get_from_yaml`) or by calling ``get_task(config={...})``, providing your own config.

    Each new environment should inherit from :class:`Task` and instantiate its members as:

    TASK_1 = None

    TASK_2 = None

    ...


    Tasks configs are loaded from ``benchmarl/conf/task``.
    """

    @staticmethod
    def associated_class() -> Type[TaskClass]:
        """
        The associated task class
        """
        raise NotImplementedError

    @classmethod
    def env_name(cls) -> str:
        """
        The name of the environment in the ``benchmarl/conf/task`` folder
        """
        return cls.associated_class().env_name()

    def get_task(self, config: Optional[Dict[str, Any]] = None) -> TaskClass:
        """
        Get the :class:`TaskClass` object associated with this enum element by passing it the task name and config.

        If no config is given, it will be loaded from ``benchmarl/conf/task/self.env_name()/self.name`` using :meth:`Task.get_from_yaml`.

        Args:
            config (dict): Optional configuration of the task.
            If not provided, the default configuration will be loaded from yaml.

        Returns:
            The :class:`TaskClass` object for the task.

        """
        if config is None:
            return self.get_from_yaml()
        return self.associated_class()(name=self.name, config=config)

    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    @staticmethod
    def _load_from_yaml(name: str) -> Dict[str, Any]:
        yaml_path = Path(__file__).parent.parent / "conf" / "task" / f"{name}.yaml"
        return _read_yaml_config(str(yaml_path.resolve()))

    def get_from_yaml(self, path: Optional[str] = None) -> TaskClass:
        """
        Load the task configuration from yaml

        Args:
            path (str, optional): The full path of the yaml file to load from. If None, it will default to
                ``benchmarl/conf/task/self.env_name()/self.name``

        Returns:
            the :class:`TaskClass` with the loaded config
        """
        environment_name = self.env_name()
        task_name = self.name.lower()
        full_name = str(Path(environment_name) / Path(task_name))
        if path is None:
            config = Task._load_from_yaml(full_name)
        else:
            config = _read_yaml_config(path)
        config = _type_check_task_config(environment_name, task_name, config)
        return self.get_task(config=config)

    ######################
    # Deprecated functions
    ######################

    @property
    def config(self):
        raise ValueError(
            "Task.config is deprecated, use Task.get_task().config instead"
        )

    def update_config(self, config: Dict[str, Any]) -> Task:
        raise ValueError(
            "Task.update_config is deprecated please use Task.get_task().config.update() instead"
        )

    def supports_continuous_actions(self) -> bool:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    def supports_discrete_actions(self) -> bool:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    def max_steps(self, env: EnvBase) -> int:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    def has_render(self, env: EnvBase) -> bool:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    def observation_spec(self, env: EnvBase) -> Composite:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    def action_spec(self, env: EnvBase) -> Composite:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    @staticmethod
    def log_info(batch: TensorDictBase) -> Dict[str, float]:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    def get_reward_sum_transform(self, env: EnvBase) -> Transform:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    def get_env_transforms(self, env: EnvBase) -> List[Transform]:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    def get_replay_buffer_transforms(self, env: EnvBase, group: str) -> List[Transform]:
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )

    @staticmethod
    def render_callback(experiment, env: EnvBase, data: TensorDictBase):
        raise ValueError(
            "Called function is deprecated is deprecated, please use Task.get_task().function() instead"
        )
