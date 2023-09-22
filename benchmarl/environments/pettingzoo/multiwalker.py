from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    task: str = MISSING
    n_walkers: int = MISSING
    shared_reward: bool = MISSING
    max_cycles: int = MISSING
