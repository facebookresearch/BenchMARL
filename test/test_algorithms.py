import importlib

import pytest
from benchmarl.algorithms import algorithm_config_registry

from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.models.common import SequenceModelConfig
from benchmarl.models.mlp import MlpConfig
from torch import nn


_has_vmas = importlib.util.find_spec("vmas") is not None


@pytest.mark.skipif(not _has_vmas, reason="VMAS not found")
@pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
@pytest.mark.parametrize("continuous", [True, False])
def test_all_algos_vmas(algo_config, continuous, experiment_config):
    task = VmasTask.BALANCE.get_from_yaml()
    model_config = SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[8], activation_class=nn.Tanh, layer_class=nn.Linear),
            MlpConfig(num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear),
        ],
        intermediate_sizes=[5],
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
