from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    task: str = MISSING
    n_walkers: int = MISSING
    shared_reward: bool = MISSING
    max_cycles: int = MISSING
    position_noise: float = MISSING
    angle_noise: float = MISSING
    forward_reward: float = MISSING
    fall_reward: float = MISSING
    terminate_reward: float = MISSING
    terminate_on_fall: bool = MISSING
    remove_on_fall: bool = MISSING
    terrain_length: float = MISSING
