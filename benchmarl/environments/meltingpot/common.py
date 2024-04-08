#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import Callable, Dict, List, Optional

import torch
from tensordict import TensorDictBase

from torchrl.data import CompositeSpec
from torchrl.envs import DoubleToFloat, DTypeCastTransform, EnvBase, Transform
from torchrl.envs.libs.meltingpot import MeltingpotEnv

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING


class MeltingPotTask(Task):
    """Enum for meltingpot tasks."""

    COMMONS_HARVEST__OPEN = None

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        return lambda: MeltingpotEnv(
            substrate=self.name.lower(),
            categorical_actions=True,
            **self.config,
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def get_transforms(self, env: EnvBase) -> List[Transform]:
        return [
            DoubleToFloat(),
            DTypeCastTransform(
                dtype_in=torch.uint8,
                dtype_out=torch.float,
                in_keys=[
                    "WORLD.RGB",
                    *[
                        (group, "observation", "RGB")
                        for group in self.group_map(env).keys()
                    ],
                ],
            ),
        ]

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            del observation_spec[group]
        if list(observation_spec.keys()) != ["WORLD.RGB"]:
            raise ValueError(
                f"More than one global state key found in observation spec {observation_spec}."
            )
        return observation_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        observation_spec = env.observation_spec.clone()
        for group_key in list(observation_spec.keys()):
            if group_key not in self.group_map(env).keys():
                del observation_spec[group_key]
            else:
                group_obs_spec = observation_spec[group_key]["observation"]
                for key in list(group_obs_spec.keys()):
                    if key != "RGB":
                        del group_obs_spec[key]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        for group_key in list(observation_spec.keys()):
            if group_key not in self.group_map(env).keys():
                del observation_spec[group_key]
            else:
                group_obs_spec = observation_spec[group_key]["observation"]
                del group_obs_spec["RGB"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return "meltingpot"

    @staticmethod
    def render_callback(experiment, env: EnvBase, data: TensorDictBase):
        return data.get("WORLD.RGB")
