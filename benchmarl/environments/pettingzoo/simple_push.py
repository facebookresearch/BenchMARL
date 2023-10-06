from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    task: str = MISSING
    max_cycles: int = MISSING
    local_ratio: float = MISSING
