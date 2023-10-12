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
    collisions: bool = MISSING
    agents_with_same_goal: int = MISSING
    observe_all_goals: bool = MISSING
    shared_rew: bool = MISSING
    split_goals: bool = MISSING
    lidar_range: float = MISSING
    agent_radius: float = MISSING
