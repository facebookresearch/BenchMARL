from .common import Task
from .vmas.balance import TaskConfig as Balance

# Environments
from .vmas.common import VmasTask

task_config_registry = {"vmas/balance": (Balance, VmasTask.BALANCE)}
