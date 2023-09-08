from benchmarl.algorithms import MaddpgConfig, MappoConfig, MasacConfig, QmixConfig
from benchmarl.benchmark import Benchmark
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.common import SequenceModelConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    experiment_config = ExperimentConfig.get_from_yaml()
    tasks = [VmasTask.BALANCE.get_from_yaml()]
    algorithm_configs = [
        MappoConfig.get_from_yaml(),
        MaddpgConfig.get_from_yaml(),
        QmixConfig.get_from_yaml(),
        MasacConfig.get_from_yaml(),
    ]
    seeds = {0}

    # Model still need to be refactored for hydra loading
    model_config = SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[64, 64]),
            MlpConfig(num_cells=[256]),
        ],
        intermediate_sizes=[128],
    )

    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks,
        seeds=seeds,
        experiment_config=experiment_config,
        model_config=model_config,
    )
    benchmark.run_sequential()
