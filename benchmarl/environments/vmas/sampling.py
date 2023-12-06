#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    shared_rew: bool = MISSING
    n_gaussians: int = MISSING
    lidar_range: float = MISSING
    cov: float = MISSING
    collisions: bool = MISSING
    spawn_same_pos: bool = MISSING
