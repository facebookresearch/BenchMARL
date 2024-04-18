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

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING


class MeltingPotTask(Task):
    """Enum for meltingpot tasks."""

    PREDATOR_PREY__ALLEY_HUNT = None
    CLEAN_UP = None
    COLLABORATIVE_COOKING__CIRCUIT = None
    FRUIT_MARKET__CONCENTRIC_RIVERS = None
    COLLABORATIVE_COOKING__FIGURE_EIGHT = None
    PAINTBALL__KING_OF_THE_HILL = None
    FACTORY_COMMONS__EITHER_OR = None
    PURE_COORDINATION_IN_THE_MATRIX__ARENA = None
    RUNNING_WITH_SCISSORS_IN_THE_MATRIX__REPEATED = None
    COLLABORATIVE_COOKING__CRAMPED = None
    RUNNING_WITH_SCISSORS_IN_THE_MATRIX__ARENA = None
    PRISONERS_DILEMMA_IN_THE_MATRIX__REPEATED = None
    TERRITORY__OPEN = None
    STAG_HUNT_IN_THE_MATRIX__REPEATED = None
    CHICKEN_IN_THE_MATRIX__REPEATED = None
    GIFT_REFINEMENTS = None
    PURE_COORDINATION_IN_THE_MATRIX__REPEATED = None
    COLLABORATIVE_COOKING__FORCED = None
    RATIONALIZABLE_COORDINATION_IN_THE_MATRIX__ARENA = None
    BACH_OR_STRAVINSKY_IN_THE_MATRIX__ARENA = None
    CHEMISTRY__TWO_METABOLIC_CYCLES_WITH_DISTRACTORS = None
    COMMONS_HARVEST__PARTNERSHIP = None
    PREDATOR_PREY__OPEN = None
    TERRITORY__ROOMS = None
    HIDDEN_AGENDA = None
    COOP_MINING = None
    DAYCARE = None
    PRISONERS_DILEMMA_IN_THE_MATRIX__ARENA = None
    TERRITORY__INSIDE_OUT = None
    BACH_OR_STRAVINSKY_IN_THE_MATRIX__REPEATED = None
    COMMONS_HARVEST__CLOSED = None
    CHEMISTRY__THREE_METABOLIC_CYCLES_WITH_PLENTIFUL_DISTRACTORS = None
    STAG_HUNT_IN_THE_MATRIX__ARENA = None
    PAINTBALL__CAPTURE_THE_FLAG = None
    COLLABORATIVE_COOKING__CROWDED = None
    ALLELOPATHIC_HARVEST__OPEN = None
    COLLABORATIVE_COOKING__RING = None
    COMMONS_HARVEST__OPEN = None
    COINS = None
    PREDATOR_PREY__ORCHARD = None
    PREDATOR_PREY__RANDOM_FOREST = None
    COLLABORATIVE_COOKING__ASYMMETRIC = None
    RATIONALIZABLE_COORDINATION_IN_THE_MATRIX__REPEATED = None
    CHEMISTRY__THREE_METABOLIC_CYCLES = None
    RUNNING_WITH_SCISSORS_IN_THE_MATRIX__ONE_SHOT = None
    CHEMISTRY__TWO_METABOLIC_CYCLES = None
    CHICKEN_IN_THE_MATRIX__ARENA = None
    BOAT_RACE__EIGHT_RACES = None
    EXTERNALITY_MUSHROOMS__DENSE = None

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        from torchrl.envs.libs.meltingpot import MeltingpotEnv

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
        return self.config.get("max_steps", 100)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def get_env_transforms(self, env: EnvBase) -> List[Transform]:
        return [DoubleToFloat()]

    def get_replay_buffer_transforms(self, env: EnvBase) -> List[Transform]:
        return [
            DTypeCastTransform(
                dtype_in=torch.uint8,
                dtype_out=torch.float,
                in_keys=[
                    "RGB",
                    *[
                        (group, "observation", "RGB")
                        for group in self.group_map(env).keys()
                    ],
                    ("next", "RGB"),
                    *[
                        ("next", group, "observation", "RGB")
                        for group in self.group_map(env).keys()
                    ],
                ],
                in_keys_inv=[],
            )
        ]

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            del observation_spec[group]
        if list(observation_spec.keys()) != ["RGB"]:
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
        return data.get("RGB")
