from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    task: str = MISSING
    N: int = MISSING
    max_cycles: int = MISSING
