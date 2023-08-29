from benchmarl.algorithms import IppoConfig, MaddpgConfig
from benchmarl.algorithms.mappo import MappoConfig
from benchmarl.benchmark import Benchmark
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.common import SequenceModelConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    tasks = [VmasTask.BALANCE]
    algorithm_configs = [
        MaddpgConfig(),
        MappoConfig(),
        IppoConfig(),
    ]
    seeds = {0}

    model_config = SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[64, 64]),
            MlpConfig(num_cells=[256]),
        ],
        intermediate_sizes=[128],
    )
    experiment_config = ExperimentConfig(n_iters=2, prefer_continuous_actions=False)

    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks,
        seeds=seeds,
        experiment_config=experiment_config,
        model_config=model_config,
    )
    benchmark.run_sequential()
