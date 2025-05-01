#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


__version__ = "1.5.0"

import importlib

import benchmarl.algorithms
import benchmarl.benchmark
import benchmarl.environments
import benchmarl.experiment
import benchmarl.models

_has_hydra = importlib.util.find_spec("hydra") is not None

if _has_hydra:

    def _load_hydra_schemas():
        from hydra.core.config_store import ConfigStore

        from benchmarl.algorithms import algorithm_config_registry
        from benchmarl.environments import _task_class_registry
        from benchmarl.experiment import ExperimentConfig

        # Create instance to load hydra schemas
        cs = ConfigStore.instance()
        # Load experiment schema
        cs.store(name="experiment_config", group="experiment", node=ExperimentConfig)
        # Load algos schemas
        for algo_name, algo_schema in algorithm_config_registry.items():
            cs.store(name=f"{algo_name}_config", group="algorithm", node=algo_schema)
        # Load task schemas
        for task_schema_name, task_schema in _task_class_registry.items():
            cs.store(name=f"{task_schema_name}_config", group="task", node=task_schema)

    _load_hydra_schemas()
