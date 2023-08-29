from typing import Iterator, Sequence, Set

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.common import ModelConfig


class Benchmark:
    def __init__(
        self,
        algorithm_configs: Sequence[AlgorithmConfig],
        model_config: ModelConfig,
        tasks: Sequence[Task],
        seeds: Set[int],
        experiment_config: ExperimentConfig,
    ):
        self.algorithm_configs = algorithm_configs
        self.tasks = tasks
        self.seeds = seeds

        self.model_config = model_config
        self.experiment_config = experiment_config

        print(f"Created benchmark with {self.n_experiments} experiments.")

    @property
    def n_experiments(self):
        return len(self.algorithm_configs) * len(self.tasks) * len(self.seeds)

    def get_experiments(self) -> Iterator[Experiment]:
        for algorithm_config in self.algorithm_configs:
            for task in self.tasks:
                for seed in self.seeds:
                    yield Experiment(
                        task=task,
                        algorithm_config=algorithm_config,
                        seed=seed,
                        model_config=self.model_config,
                        config=self.experiment_config,
                    )

    def run_sequential(self):
        for i, experiment in enumerate(self.get_experiments()):
            print(f"\nRunning experiment {i+1}/{self.n_experiments}.\n")
            experiment.run()
