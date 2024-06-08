#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_blue_agents: int = MISSING
    n_red_agents: int = MISSING
    ball_mass: float = MISSING
    disable_ai_red: bool = MISSING
    ai_strength: float = MISSING
    observe_teammates: bool = MISSING
