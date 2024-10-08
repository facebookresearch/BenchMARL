#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    minimap_mode: bool = MISSING
    step_reward: float = MISSING
    attack_penalty: float = MISSING
    dead_penalty: float = MISSING
    attack_food_reward: float = MISSING
    max_cycles: int = MISSING
    extra_features: bool = MISSING
