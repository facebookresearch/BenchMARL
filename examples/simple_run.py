from benchmarl.algorithms.mappo import MappoConfig
from benchmarl.benchmark import Benchmark
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.common import SequenceModelConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    tasks = [VmasTask.BALANCE.get_from_yaml()]
    algorithm_configs = [MappoConfig.get_from_yaml()]
    seeds = {0}

    model_config = SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[64, 64]),
            MlpConfig(num_cells=[256]),
        ],
        intermediate_sizes=[128],
    )
    experiment_config = ExperimentConfig.get_from_yaml()

    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks,
        seeds=seeds,
        experiment_config=experiment_config,
        model_config=model_config,
    )
    benchmark.run_sequential()
