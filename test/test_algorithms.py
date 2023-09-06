import pytest

from benchmarl.algorithms import all_algorithm_configs
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.common import SequenceModelConfig
from benchmarl.models.mlp import MlpConfig


@pytest.mark.parametrize("algo_config", all_algorithm_configs)
@pytest.mark.parametrize("continuous", [True, False])
def test_all_algos_balance(algo_config, continuous):
    task = VmasTask.BALANCE
    model_config = SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[8]),
            MlpConfig(num_cells=[4]),
        ],
        intermediate_sizes=[5],
    )
    experiment_config = ExperimentConfig(
        n_iters=2, prefer_continuous_actions=continuous
    )
    experiment = Experiment(
        algorithm_config=algo_config(),
        model_config=model_config,
        seed=0,
        config=experiment_config,
        task=task,
    )
    experiment.run()
