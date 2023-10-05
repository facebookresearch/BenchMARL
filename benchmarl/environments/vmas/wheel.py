from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    line_length: float = MISSING
    line_mass: float = MISSING
    desired_velocity: float = MISSING
