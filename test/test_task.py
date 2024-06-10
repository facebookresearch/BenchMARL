#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest

from benchmarl.environments import Task, task_config_registry
from benchmarl.hydra_config import load_task_config_from_hydra
from hydra import compose, initialize


@pytest.mark.parametrize("task_name", task_config_registry.keys())
def test_loading_tasks(task_name):
    with initialize(version_base=None, config_path="../benchmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=mappo",
                f"task={task_name}",
            ],
            return_hydra_config=True,
        )
        task_name_hydra = cfg.hydra.runtime.choices.task
        task: Task = load_task_config_from_hydra(cfg.task, task_name=task_name_hydra)
        assert task == task_config_registry[task_name].get_from_yaml()
