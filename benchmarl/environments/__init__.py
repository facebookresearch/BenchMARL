#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import _get_task_config_class, Task

from .magent.common import MAgentTask
from .meltingpot.common import MeltingPotTask
from .pettingzoo.common import PettingZooTask
from .smacv2.common import Smacv2Task
from .vmas.common import VmasTask

# The enum classes for the environments available.
# This is the only object in this file you need to modify when adding a new environment.
tasks = [VmasTask, Smacv2Task, PettingZooTask, MeltingPotTask, MAgentTask]

# This is a registry mapping "envname/task_name" to the EnvNameTask.TASK_NAME enum
# It is used by automatically load task enums from yaml files.
# It is populated automatically, do not modify.
task_config_registry = {}

# This is a registry mapping "envname_taskname" to the TaskConfig python dataclass of the task.
# It is used by hydra to validate loaded configs.
# You will see the "envname_taskname" strings in the hydra defaults at the top of yaml files.
# This is optional and, if a task does not possess an associated TaskConfig, this entry will be simply skipped.
# It is populated automatically, do not modify.
_task_class_registry = {}

# Automatic population of registries
for env in tasks:
    env_config_registry = {}
    environemnt_name = env.env_name()
    for task in env:
        task_name = task.name.lower()
        full_task_name = f"{environemnt_name}/{task_name}"
        env_config_registry[full_task_name] = task

        task_config_class = _get_task_config_class(environemnt_name, task_name)
        if task_config_class is not None:
            _task_class_registry[full_task_name.replace("/", "_")] = task_config_class
    task_config_registry.update(env_config_registry)
