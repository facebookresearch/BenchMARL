#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import importlib

_has_hydra = importlib.util.find_spec("hydra") is not None

if _has_hydra:

    def load_hydra_schemas():
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
            cs.store(name=task_schema_name, group="task", node=task_schema)

    load_hydra_schemas()
