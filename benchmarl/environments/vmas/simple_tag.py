#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    num_good_agents: int = MISSING
    num_adversaries: int = MISSING
    num_landmarks: int = MISSING
    shape_agent_rew: bool = MISSING
    shape_adversary_rew: bool = MISSING
    agents_share_rew: bool = MISSING
    adversaries_share_rew: bool = MISSING
    observe_same_team: bool = MISSING
    observe_pos: bool = MISSING
    observe_vel: bool = MISSING
    bound: float = MISSING
    respawn_at_catch: bool = MISSING
