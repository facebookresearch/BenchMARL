#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_passages: int = MISSING
    fixed_passage: bool = MISSING
    joint_length: float = MISSING
    random_start_angle: bool = MISSING
    random_goal_angle: bool = MISSING
    observe_joint_angle: bool = MISSING
    asym_package: bool = MISSING
    mass_ratio: float = MISSING
    mass_position: float = MISSING
