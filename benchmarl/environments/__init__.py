from .common import Task
from .vmas.balance import TaskConfig as Balance, TaskConfig as BalanceConfig

# Environments
from .vmas.common import VmasTask

task_config_registry = {"vmas/balance": VmasTask.BALANCE}

_task_class_registry = {"vmas_balance_config": BalanceConfig}
