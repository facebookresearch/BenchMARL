from .common import Task

from .pettingzoo.common import PettingZooTask
from .pettingzoo.multiwalker import TaskConfig as MultiwalkerConfig
from .pettingzoo.simple_tag import TaskConfig as SimpleTagConfig

from .smacv2.common import Smacv2Task
from .vmas.balance import TaskConfig as BalanceConfig
from .vmas.common import VmasTask
from .vmas.navigation import TaskConfig as NavigationConfig
from .vmas.sampling import TaskConfig as SamplingConfig


task_config_registry = {
    "vmas/balance": VmasTask.BALANCE,
    "vmas/sampling": VmasTask.SAMPLING,
    "vmas/navigation": VmasTask.NAVIGATION,
    "smacv2/protoss_5_vs_5": Smacv2Task.protoss_5_vs_5,
    "pettingzoo/multiwalker": PettingZooTask.MULTIWALKER,
    "pettingzoo/simple_tag": PettingZooTask.SIMPLE_TAG,
}


_task_class_registry = {
    "vmas_balance_config": BalanceConfig,
    "vmas_sampling_config": SamplingConfig,
    "vmas_navigation_config": NavigationConfig,
    "pettingzoo_multiwalker_config": MultiwalkerConfig,
    "pettingzoo_simple_tag_config": SimpleTagConfig,
}
