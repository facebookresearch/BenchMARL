from .common import Task
from .pettingzoo.common import PettingZooTask
from .smacv2.common import Smacv2Task
from .vmas.common import VmasTask

task_config_registry = {}
for env in [VmasTask, Smacv2Task, PettingZooTask]:
    env_config_registry = {
        f"{env.env_name()}/{task.name.lower()}": task for task in env
    }
    task_config_registry.update(env_config_registry)


from .pettingzoo.multiwalker import TaskConfig as MultiwalkerConfig
from .pettingzoo.simple_tag import TaskConfig as SimpleTagConfig
from .vmas.balance import TaskConfig as BalanceConfig
from .vmas.navigation import TaskConfig as NavigationConfig
from .vmas.sampling import TaskConfig as SamplingConfig
from .vmas.transport import TaskConfig as TransportConfig
from .vmas.wheel import TaskConfig as WheelConfig

_task_class_registry = {
    "vmas_balance_config": BalanceConfig,
    "vmas_sampling_config": SamplingConfig,
    "vmas_navigation_config": NavigationConfig,
    "vmas_transport_config": TransportConfig,
    "vmas_wheel_config": WheelConfig,
    "pettingzoo_multiwalker_config": MultiwalkerConfig,
    "pettingzoo_simple_tag_config": SimpleTagConfig,
}
