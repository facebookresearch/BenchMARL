from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    random_package_pos_on_line: bool = MISSING
    package_mass: float = MISSING
