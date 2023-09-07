import importlib
import os
import os.path as osp
import pathlib
from enum import Enum
from typing import Any, Dict, List, Optional

from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase
from utils import read_yaml_file


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
    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, config: Dict):
        self.config = config

    def update_config(self, config: Dict[str, Any]):
        self.config.update(config)
        return self

    def get_env(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
    ) -> EnvBase:
        raise NotImplementedError

    def supports_continuous_actions(self) -> bool:
        raise NotImplementedError

    def supports_discrete_actions(self) -> bool:
        raise NotImplementedError

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        raise NotImplementedError

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        raise NotImplementedError

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        raise NotImplementedError

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        raise NotImplementedError

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        raise NotImplementedError

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        raise NotImplementedError

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}: (config={self.config})"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def get_from_yaml(environment_name: str, task_name: str) -> Dict[str, Any]:
        yaml_path = (
            pathlib.Path(__file__).parent.parent
            / "conf"
            / "task"
            / environment_name
            / f"{task_name.lower()}.yaml"
        )
        config = read_yaml_file(str(yaml_path.resolve()))
        del config["defaults"]
        config = _load_config(task_name, config)
        return config
