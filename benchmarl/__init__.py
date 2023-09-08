from hydra.core.config_store import ConfigStore

from benchmarl.algorithms import algorithm_config_registry
from benchmarl.environments import _task_class_registry
from benchmarl.experiment import ExperimentConfig


def load_hydra_schemas():
    # Create instance to load hydra schemas
    cs = ConfigStore.instance()
    # Load experiment schema
    cs.store(name="experiment_config", group="experiment", node=ExperimentConfig)
    # Load algos schemas
    for algo_name, algo_schema in algorithm_config_registry.items():
        cs.store(name=f"{algo_name}_config", group="algorithm", node=algo_schema)
    # Load rask schemas
    for task_schema_name, task_schema in _task_class_registry.items():
        cs.store(name=task_schema_name, group="task", node=task_schema)


load_hydra_schemas()
