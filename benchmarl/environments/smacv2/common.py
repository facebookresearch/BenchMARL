#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import Callable, Dict, List, Optional

import torch
from tensordict import TensorDictBase
from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase
from torchrl.envs.libs.smacv2 import SMACv2Env

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING


class Smacv2Task(Task):
    PROTOSS_5_VS_5 = None
    PROTOSS_10_VS_10 = None
    PROTOSS_10_VS_11 = None
    PROTOSS_20_VS_20 = None
    PROTOSS_20_VS_23 = None
    TERRAN_5_VS_5 = None
    TERRAN_10_VS_10 = None
    TERRAN_10_VS_11 = None
    TERRAN_20_VS_20 = None
    TERRAN_20_VS_23 = None
    ZERG_5_VS_5 = None
    ZERG_10_VS_10 = None
    ZERG_10_VS_11 = None
    ZERG_20_VS_20 = None
    ZERG_20_VS_23 = None

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        return lambda: SMACv2Env(
            categorical_actions=True, seed=seed, device=device, **self.config
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return env.episode_limit

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        del observation_spec["info"]
        del observation_spec["agents"]
        return observation_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        del observation_spec["info"]
        del observation_spec["state"]
        del observation_spec[("agents", "observation")]
        return observation_spec

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        observation_spec = env.observation_spec.clone()
        del observation_spec["info"]
        del observation_spec["state"]
        del observation_spec[("agents", "action_mask")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        del observation_spec["state"]
        del observation_spec["agents"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.input_spec["full_action_spec"]

    @staticmethod
    def log_info(batch: TensorDictBase) -> Dict[str, float]:
        done = batch.get(("next", "done")).squeeze(-1)
        return {
            "collection/info/win_rate": batch.get(("next", "info", "battle_won"))[done]
            .to(torch.float)
            .mean()
            .item(),
            "collection/info/episode_limit_rate": batch.get(
                ("next", "info", "episode_limit")
            )[done]
            .to(torch.float)
            .mean()
            .item(),
        }

    @staticmethod
    def env_name() -> str:
        return "smacv2"
