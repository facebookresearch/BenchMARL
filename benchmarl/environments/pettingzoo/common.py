#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooEnv

from benchmarl.environments.common import Task, TaskClass

from benchmarl.utils import DEVICE_TYPING


class PettingZooClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        if self.supports_continuous_actions() and self.supports_discrete_actions():
            config.update({"continuous_actions": continuous_actions})
        return lambda: PettingZooEnv(
            categorical_actions=True,
            device=device,
            seed=seed,
            parallel=True,
            return_state=self.has_state(),
            render_mode="rgb_array",
            **config
        )

    def supports_continuous_actions(self) -> bool:
        if self.name in {
            "MULTIWALKER",
            "WATERWORLD",
            "SIMPLE_TAG",
            "SIMPLE_ADVERSARY",
            "SIMPLE_CRYPTO",
            "SIMPLE_PUSH",
            "SIMPLE_REFERENCE",
            "SIMPLE_SPEAKER_LISTENER",
            "SIMPLE_SPREAD",
            "SIMPLE_TAG",
            "SIMPLE_WORLD_COMM",
        }:
            return True
        return False

    def supports_discrete_actions(self) -> bool:
        if self.name in {
            "SIMPLE_TAG",
            "SIMPLE_ADVERSARY",
            "SIMPLE_CRYPTO",
            "SIMPLE_PUSH",
            "SIMPLE_REFERENCE",
            "SIMPLE_SPEAKER_LISTENER",
            "SIMPLE_SPREAD",
            "SIMPLE_TAG",
            "SIMPLE_WORLD_COMM",
        }:
            return True
        return False

    def has_state(self) -> bool:
        if self.name in {
            "SIMPLE_TAG",
            "SIMPLE_ADVERSARY",
            "SIMPLE_CRYPTO",
            "SIMPLE_PUSH",
            "SIMPLE_REFERENCE",
            "SIMPLE_SPEAKER_LISTENER",
            "SIMPLE_SPREAD",
            "SIMPLE_TAG",
            "SIMPLE_WORLD_COMM",
        }:
            return True
        return False

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_cycles"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        if "state" in env.observation_spec:
            return Composite({"state": env.observation_spec["state"].clone()})
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "action_mask":
                    del group_obs_spec[key]
            if group_obs_spec.is_empty():
                del observation_spec[group]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        if observation_spec.is_empty():
            return None
        return observation_spec

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return "pettingzoo"


class PettingZooTask(Task):
    """Enum for PettingZoo tasks."""

    MULTIWALKER = None
    WATERWORLD = None
    SIMPLE_ADVERSARY = None
    SIMPLE_CRYPTO = None
    SIMPLE_PUSH = None
    SIMPLE_REFERENCE = None
    SIMPLE_SPEAKER_LISTENER = None
    SIMPLE_SPREAD = None
    SIMPLE_TAG = None
    SIMPLE_WORLD_COMM = None

    @staticmethod
    def associated_class():
        return PettingZooClass
