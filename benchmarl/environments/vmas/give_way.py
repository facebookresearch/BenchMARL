#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    mirror_passage: bool = MISSING
    observe_rel_pos: bool = MISSING
    done_on_completion: bool = MISSING
    final_reward: float = MISSING
