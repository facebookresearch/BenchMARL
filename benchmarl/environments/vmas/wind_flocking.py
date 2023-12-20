#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    dist_shaping_factor: float = MISSING
    rot_shaping_factor: float = MISSING
    vel_shaping_factor: float = MISSING
    pos_shaping_factor: float = MISSING
    energy_shaping_factor: float = MISSING
    wind_shaping_factor: float = MISSING
    wind: float = MISSING
    cover_angle_tolerance: float = MISSING
    horizon: int = MISSING
    observe_rel_pos: bool = MISSING
    observe_rel_vel: bool = MISSING
    observe_pos: bool = MISSING
    desired_vel: float = MISSING
