import importlib
import os
import os.path as osp
from enum import Enum
from typing import Dict, List, Optional

from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase


def load_config(name: str):
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
    config = module.TaskConfig().__dict__
    return config


class Task(Enum):
    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, config: Dict):
        self.config = config

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
