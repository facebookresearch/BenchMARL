from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    task: str = MISSING
    num_good: int = MISSING
    num_adversaries: int = MISSING
    num_obstacles: int = MISSING
    max_cycles: int = MISSING
