from dataclasses import dataclass


@dataclass
class TaskConfig:
    max_steps: int = 100
