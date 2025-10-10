from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    task: str = MISSING 
    parallel: bool = MISSING
    max_cycles: int = MISSING
    num_agents: int = MISSING
    num_bases: int = MISSING
    num_messages: int = MISSING
    random_base_pose: bool = MISSING 
    antenna_used: bool = MISSING
    observe_self: bool = MISSING
