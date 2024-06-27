#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING
from typing import Optional


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    pos_shaping_factor: float = MISSING
    final_reward: float = MISSING
    agent_collision_penalty: float = MISSING
    v_range: float = MISSING
    a_range: float = MISSING
    edge_radius: Optional[float] = MISSING
