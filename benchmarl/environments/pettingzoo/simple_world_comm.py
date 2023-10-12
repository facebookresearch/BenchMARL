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
    num_good: int = MISSING
    num_adversaries: int = MISSING
    num_obstacles: int = MISSING
    num_food: int = MISSING
    num_forests: int = MISSING
