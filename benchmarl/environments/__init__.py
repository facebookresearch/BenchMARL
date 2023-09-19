from .common import Task
from .vmas.balance import TaskConfig as BalanceConfig

# Environments
from .vmas.common import VmasTask
from .vmas.navigation import TaskConfig as NavigationConfig
from .vmas.sampling import TaskConfig as SamplingConfig


task_config_registry = {
    "vmas/balance": VmasTask.BALANCE,
    "vmas/sampling": VmasTask.SAMPLING,
    "vmas/navigation": VmasTask.NAVIGATION,
}


_task_class_registry = {
    "vmas_balance_config": BalanceConfig,
    "vmas_sampling_config": SamplingConfig,
    "vmas_navigation_config": NavigationConfig,
}
