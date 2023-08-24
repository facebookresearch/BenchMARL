import sys
from dataclasses import asdict
from enum import Enum
from typing import Dict, Optional
from torchrl.envs import EnvBase
import importlib
import os
import os.path as osp


def load_config(name: str):
    if not name.endswith(".py"):
        name += ".py"

    pathname = None
    for dirpath, dirnames, filenames in os.walk(osp.dirname(__file__)):
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
    config = asdict(module.TaskConfig())
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
