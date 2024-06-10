#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import Task
from .meltingpot.common import MeltingPotTask
from .pettingzoo.common import PettingZooTask
from .smacv2.common import Smacv2Task
from .vmas.common import VmasTask

# This is a registry mapping "envname/task_name" to the EnvNameTask.TASK_NAME enum
# It is used by automatically load task enums from yaml files
task_config_registry = {}
for env in [VmasTask, Smacv2Task, PettingZooTask, MeltingPotTask]:
    env_config_registry = {}
    environemnt_name = env.env_name()
    for task in env:
        task_name = task.name.lower()
        full_task_name = f"{environemnt_name}/{task_name}"
        env_config_registry[full_task_name] = task
    task_config_registry.update(env_config_registry)


from .pettingzoo.multiwalker import TaskConfig as MultiwalkerConfig
from .pettingzoo.simple_adverasary import TaskConfig as SimpleAdversaryConfig
from .pettingzoo.simple_crypto import TaskConfig as SimpleCryptoConfig
from .pettingzoo.simple_push import TaskConfig as SimplePushConfig
from .pettingzoo.simple_reference import TaskConfig as SimpleReferenceConfig
from .pettingzoo.simple_speaker_listener import (
    TaskConfig as SimpleSpeakerListenerConfig,
)
from .pettingzoo.simple_spread import TaskConfig as SimpleSpreadConfig
from .pettingzoo.simple_tag import TaskConfig as SimpleTagConfig
from .pettingzoo.simple_world_comm import TaskConfig as SimpleWorldComm
from .pettingzoo.waterworld import TaskConfig as WaterworldConfig

from .vmas.balance import TaskConfig as BalanceConfig
from .vmas.ball_passage import TaskConfig as BallPassageConfig
from .vmas.ball_trajectory import TaskConfig as BallTrajectoryConfig
from .vmas.buzz_wire import TaskConfig as BuzzWireConfig
from .vmas.discovery import TaskConfig as DiscoveryConfig
from .vmas.dispersion import TaskConfig as DispersionConfig
from .vmas.dropout import TaskConfig as DropoutConfig
from .vmas.flocking import TaskConfig as FlockingConfig
from .vmas.give_way import TaskConfig as GiveWayConfig
from .vmas.joint_passage import TaskConfig as JointPassageConfig
from .vmas.joint_passage_size import TaskConfig as JointPassageSizeConfig
from .vmas.multi_give_way import TaskConfig as MultiGiveWayConfig
from .vmas.navigation import TaskConfig as NavigationConfig
from .vmas.passage import TaskConfig as PassageConfig
from .vmas.reverse_transport import TaskConfig as ReverseTransportConfig
from .vmas.sampling import TaskConfig as SamplingConfig
from .vmas.simple_adverasary import TaskConfig as VmasSimpleAdversaryConfig
from .vmas.simple_crypto import TaskConfig as VmasSimpleCryptoConfig
from .vmas.simple_push import TaskConfig as VmasSimplePushConfig
from .vmas.simple_reference import TaskConfig as VmasSimpleReferenceConfig
from .vmas.simple_speaker_listener import TaskConfig as VmasSimpleSpeakerListenerConfig
from .vmas.simple_spread import TaskConfig as VmasSimpleSpreadConfig
from .vmas.simple_tag import TaskConfig as VmasSimpleTagConfig
from .vmas.simple_world_comm import TaskConfig as VmasSimpleWorldComm
from .vmas.transport import TaskConfig as TransportConfig
from .vmas.wheel import TaskConfig as WheelConfig
from .vmas.wind_flocking import TaskConfig as WindFlockingConfig

# This is a registry mapping task config schemas names to their python dataclass
# It is used by hydra to validate loaded configs.
# You will see the "envname_taskname_config" strings in the hydra defaults at the top of yaml files.
# This feature is optional.
_task_class_registry = {
    "vmas_balance_config": BalanceConfig,
    "vmas_sampling_config": SamplingConfig,
    "vmas_navigation_config": NavigationConfig,
    "vmas_transport_config": TransportConfig,
    "vmas_reverse_transport_config": ReverseTransportConfig,
    "vmas_wheel_config": WheelConfig,
    "vmas_dispersion_config": DispersionConfig,
    "vmas_give_way_config": GiveWayConfig,
    "vmas_multi_give_way_config": MultiGiveWayConfig,
    "vmas_passage_config": PassageConfig,
    "vmas_joint_passage_config": JointPassageConfig,
    "vmas_joint_passage_size_config": JointPassageSizeConfig,
    "vmas_ball_passage_config": BallPassageConfig,
    "vmas_buzz_wire_config": BuzzWireConfig,
    "vmas_ball_trajectory_config": BallTrajectoryConfig,
    "vmas_flocking_config": FlockingConfig,
    "vmas_wind_flocking_config": WindFlockingConfig,
    "vmas_dropout_config": DropoutConfig,
    "vmas_discovery_config": DiscoveryConfig,
    "vmas_simple_adversary_config": VmasSimpleAdversaryConfig,
    "vmas_simple_crypto_config": VmasSimpleCryptoConfig,
    "vmas_simple_push_config": VmasSimplePushConfig,
    "vmas_simple_reference_config": VmasSimpleReferenceConfig,
    "vmas_simple_speaker_listener_config": VmasSimpleSpeakerListenerConfig,
    "vmas_simple_spread_config": VmasSimpleSpreadConfig,
    "vmas_simple_tag_config": VmasSimpleTagConfig,
    "vmas_simple_world_comm_config": VmasSimpleWorldComm,
    "pettingzoo_multiwalker_config": MultiwalkerConfig,
    "pettingzoo_waterworld_config": WaterworldConfig,
    "pettingzoo_simple_adversary_config": SimpleAdversaryConfig,
    "pettingzoo_simple_crypto_config": SimpleCryptoConfig,
    "pettingzoo_simple_push_config": SimplePushConfig,
    "pettingzoo_simple_reference_config": SimpleReferenceConfig,
    "pettingzoo_simple_speaker_listener_config": SimpleSpeakerListenerConfig,
    "pettingzoo_simple_spread_config": SimpleSpreadConfig,
    "pettingzoo_simple_tag_config": SimpleTagConfig,
    "pettingzoo_simple_world_comm_config": SimpleWorldComm,
}
