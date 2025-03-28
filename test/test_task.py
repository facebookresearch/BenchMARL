#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import contextlib

import pytest
from benchmarl.environments import _task_class_registry, task_config_registry
from benchmarl.hydra_config import load_task_config_from_hydra
from hydra import compose, initialize


@pytest.mark.parametrize("task_name", task_config_registry.keys())
def test_loading_tasks(task_name):
    task_dataclasses_names = list(_task_class_registry.keys())
    config_task_name = task_name.replace("/", "_")
    task_has_dataclass = False
    for task_dataclass_name in task_dataclasses_names:
        if config_task_name in task_dataclass_name:
            task_has_dataclass = True
            break

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
        assert task_name_hydra == task_name

        warn_message = "TaskConfig python dataclass not found, task is being loaded without type checks"

        with (
            pytest.warns(match=warn_message)
            if not task_has_dataclass
            else contextlib.nullcontext()
        ):
            task = load_task_config_from_hydra(cfg.task, task_name=task_name_hydra)

        with (
            pytest.warns(match=warn_message)
            if not task_has_dataclass
            else contextlib.nullcontext()
        ):
            task_from_yaml = task_config_registry[task_name].get_from_yaml()

        assert task == task_from_yaml
