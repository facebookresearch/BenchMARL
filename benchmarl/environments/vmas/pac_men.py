#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING
from typing import Tuple


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    shared_rew: bool = MISSING
    corridors_length: float = MISSING
    observe_position: bool = MISSING
    spawn_same_pos: bool = MISSING
    lrud_ratio: Tuple[float] = MISSING
