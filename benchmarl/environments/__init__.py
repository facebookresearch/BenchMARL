from .common import Task
from .vmas.balance import TaskConfig as BalanceConfig

# Environments
from .vmas.common import VmasTask
from .vmas.sampling import TaskConfig as SamplingConfig

task_config_registry = {
    "vmas/balance": VmasTask.BALANCE,
    "vmas/sampling": VmasTask.SAMPLING,
}


_task_class_registry = {
    "vmas_balance_config": BalanceConfig,
    "vmas_sampling_config": SamplingConfig,
}
