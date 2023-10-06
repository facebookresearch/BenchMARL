from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    task: str = MISSING
    n_walkers: int = MISSING
    shared_reward: bool = MISSING
    max_cycles: int = MISSING
    x_size: int = MISSING
    y_size: int = MISSING
    n_evaders: int = MISSING
    n_pursuers: int = MISSING
    obs_range: int = MISSING
    n_catch: int = MISSING
    freeze_evaders: bool = MISSING
    tag_reward: float = MISSING
    catch_reward: float = MISSING
    urgency_reward: float = MISSING
    surround: bool = MISSING
    constraint_window: float = MISSING
