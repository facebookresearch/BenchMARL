from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    n_packages: int = MISSING
    package_width: float = MISSING
    package_length: float = MISSING
    package_mass: float = MISSING
