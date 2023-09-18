import pathlib

import pytest

from benchmarl.algorithms import algorithm_config_registry

from benchmarl.environments import Smacv2Task, VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.common import SequenceModelConfig
from benchmarl.models.mlp import MlpConfig
from torch import nn


@pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
@pytest.mark.parametrize("continuous", [True, False])
def test_all_algos_balance(algo_config, continuous):
    task = VmasTask.BALANCE.get_from_yaml()
    model_config = SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[8], activation_class=nn.Tanh, layer_class=nn.Linear),
            MlpConfig(num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear),
        ],
        intermediate_sizes=[5],
    )
    experiment_config: ExperimentConfig = ExperimentConfig.get_from_yaml(
        str(
            pathlib.Path(__file__).parent
            / "conf"
            / "experiment"
            / "base_experiment.yaml"
        )
    )
    experiment_config.prefer_continuous_actions = continuous

    experiment = Experiment(
        algorithm_config=algo_config.get_from_yaml(),
        model_config=model_config,
        seed=0,
        config=experiment_config,
        task=task,
    )
    experiment.run()


@pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
def test_all_algos_protos(algo_config, continuous=False):
    task = Smacv2Task.protoss_5_vs_5.get_from_yaml()
    model_config = SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[8], activation_class=nn.Tanh, layer_class=nn.Linear),
            MlpConfig(num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear),
        ],
        intermediate_sizes=[5],
    )
    experiment_config: ExperimentConfig = ExperimentConfig.get_from_yaml(
        str(
            pathlib.Path(__file__).parent
            / "conf"
            / "experiment"
            / "base_experiment.yaml"
        )
    )
    experiment_config.prefer_continuous_actions = continuous

    experiment = Experiment(
        algorithm_config=algo_config.get_from_yaml(),
        model_config=model_config,
        seed=0,
        config=experiment_config,
        task=task,
    )
    experiment.run()


# @pytest.mark.parametrize("algo_config", algorithm_config_registry.keys())
# def test_all_algos_hydra(algo_config):
#     with initialize(version_base=None, config_path="../benchmarl/conf"):
#         cfg = compose(
#             config_name="config",
#             overrides=[
#                 f"algorithm={algo_config}",
#                 "task=vmas/balance",
#                 "model.num_cells=[3]",
#                 "experiment.loggers=[]",
#             ],
#             return_hydra_config=True,
#         )
#         task_name = cfg.hydra.runtime.choices.task
#         experiment = load_experiment_from_hydra_config(cfg, task_name=task_name)
#         experiment.run()
