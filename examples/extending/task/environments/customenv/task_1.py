#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    n_borks: int = MISSING
    win_on_dork: bool = MISSING
