#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooWrapper

from benchmarl.environments.common import Task, TaskClass

from benchmarl.utils import DEVICE_TYPING


class MAgentClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)

        return lambda: PettingZooWrapper(
            env=self.__get_env(config),
            return_state=True,
            seed=seed,
            done_on_any=False,
            use_mask=False,
            device=device,
        )

    def __get_env(self, config) -> EnvBase:
        try:
            from magent2.environments import (
                adversarial_pursuit_v4,
                # battle_v4,
                # battlefield_v5,
                # combined_arms_v6,
                # gather_v5,
                # tiger_deer_v4
            )
        except ImportError:
            raise ImportError(
                "Module `magent2` not found, install it using `pip install magent2`"
            )

        envs = {
            "ADVERSARIAL_PURSUIT": adversarial_pursuit_v4,
            # "BATTLE": battle_v4,
            # "BATTLEFIELD": battlefield_v5,
            # "COMBINED_ARMS": combined_arms_v6,
            # "GATHER": gather_v5,
            # "TIGER_DEER": tiger_deer_v4
        }
        if self.name not in envs:
            raise Exception(f"{self.name} is not an environment of MAgent2")
        return envs[self.name].parallel_env(**config, render_mode="rgb_array")

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_state(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_cycles"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return Composite({"state": env.observation_spec["state"].clone()})

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "action_mask":
                    del group_obs_spec[key]
            if group_obs_spec.is_empty():
                del observation_spec[group]
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
        del observation_spec["state"]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        del observation_spec["state"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return "magent"


class MAgentTask(Task):
    """Enum for MAgent2 tasks."""

    ADVERSARIAL_PURSUIT = None
    # BATTLE = None
    # BATTLEFIELD = None
    # COMBINED_ARMS = None
    # GATHER = None
    # TIGER_DEER = None

    @staticmethod
    def associated_class():
        return MAgentClass
