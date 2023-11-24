#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    task: str = MISSING
    max_cycles: int = MISSING
    n_pursuers: int = MISSING
    n_evaders: int = MISSING
    n_poisons: int = MISSING
    n_obstacles: int = MISSING
    n_coop: int = MISSING
    n_sensors: int = MISSING
    sensor_range: float = MISSING
    radius: float = MISSING
    obstacle_radius: float = MISSING
    pursuer_max_accel: float = MISSING
    pursuer_speed: float = MISSING
    evader_speed: float = MISSING
    poison_speed: float = MISSING
    poison_reward: float = MISSING
    food_reward: float = MISSING
    encounter_reward: float = MISSING
    thrust_penalty: float = MISSING
    local_ratio: float = MISSING
    speed_features: bool = MISSING
