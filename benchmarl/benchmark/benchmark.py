#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import Iterator, Optional, Sequence, Set, Union

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task, TaskClass
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.common import ModelConfig


class Benchmark:
    """A benchmark.

    Benchmarks are collections of experiments to compare.

    Args:
        algorithm_configs (sequence of AlgorithmConfig): the algorithms to benchmark
        model_config (ModelConfig): the config of the policy model
        tasks (sequence of Task):  the tasks to benchmark
        seeds (set of int): the seeds for the benchmark
        experiment_config (ExperimentConfig): the experiment config
        critic_model_config (ModelConfig, optional): the config of the critic model. Defaults to model_config

    """

    def __init__(
        self,
        algorithm_configs: Sequence[AlgorithmConfig],
        model_config: ModelConfig,
        tasks: Sequence[Union[Task, TaskClass]],
        seeds: Set[int],
        experiment_config: ExperimentConfig,
        critic_model_config: Optional[ModelConfig] = None,
    ):
        self.algorithm_configs = algorithm_configs
        self.tasks = tasks
        self.seeds = seeds

        self.model_config = model_config
        self.critic_model_config = (
            critic_model_config if critic_model_config is not None else model_config
        )
        self.experiment_config = experiment_config

        print(f"Created benchmark with {self.n_experiments} experiments.")

    @property
    def n_experiments(self):
        """The number of experiments in the benchmark."""
        return len(self.algorithm_configs) * len(self.tasks) * len(self.seeds)

    def get_experiments(self) -> Iterator[Experiment]:
        """Yields one experiment at a time"""
        for algorithm_config in self.algorithm_configs:
            for task in self.tasks:
                for seed in self.seeds:
                    yield Experiment(
                        task=task,
                        algorithm_config=algorithm_config,
                        seed=seed,
                        model_config=self.model_config,
                        critic_model_config=self.critic_model_config,
                        config=self.experiment_config,
                    )

    def run_sequential(self):
        """Run all the experiments in the benchmark in a sequence."""
        for i, experiment in enumerate(self.get_experiments()):
            print(f"\nRunning experiment {i+1}/{self.n_experiments}.\n")
            try:
                experiment.run()
            except KeyboardInterrupt as interrupt:
                print("\n\nBenchmark was closed gracefully\n\n")
                experiment.close()
                raise interrupt
