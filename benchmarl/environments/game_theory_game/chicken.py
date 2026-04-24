from dataclasses import dataclass, MISSING

# These parameters will be vailable in self.config
# Example: config = copy.deepcopy(self.config)

@dataclass
class TaskConfig:
    game_name = "chicken"