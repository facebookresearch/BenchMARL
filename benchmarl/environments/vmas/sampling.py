from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    shared_rew: bool = MISSING
    n_gaussians: int = MISSING
    lidar_range: float = MISSING
