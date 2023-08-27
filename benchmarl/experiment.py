from dataclasses import dataclass, asdict

from benchmarl.environments import Task, VmasTask
from benchmarl.models.common import ModelConfig
from benchmarl.algorithms.common import AlgorithmConfig


class Experiment:
    def __init__(
        self,
        task: Task,
        algorithm_config: AlgorithmConfig,
        model_config: ModelConfig,
        lr: float,
        n_optimizer_steps: int,
        num_vectorized_envs: int,
        prefer_continuous_actions: bool = True,
    ):
        self.task = task
        self.algorithm_config = algorithm_config
        self.model_config = model_config

        self.lr = lr
        self.n_optimizer_steps = n_optimizer_steps
        self.num_vectorized_envs = num_vectorized_envs

        self.prefer_continuous_actions = prefer_continuous_actions

    def _setup_task(self):
        self.continuous_actions = self.prefer_continuous_actions

        if (
            self.task.supports_continuous_actions()
            and AlgorithmConfig.associated_class().supports_continuous_actions()
            and self.prefer_continuous_actions
        ):
            self.continuous_actions = True
        elif (
            self.task.supports_discrete_actions()
            and AlgorithmConfig.associated_class().supports_discrete_actions()
        ):
            self.continuous_actions = False

        self.env = self.task.get_env(
            num_envs=self.num_vectorized_envs,
            continuous_actions=self.continuous_actions,
        )

    def run(self):
        pass


@dataclass
class ExperimentConfig:
    # These are the kwargs of benchmarl.Experiment

    task: Task
    algorithm_config: AlgorithmConfig
    model_config: ModelConfig

    lr: float = 3e-5
    n_optimizer_steps: int = 10
    num_vectorized_envs: int = 1

    def get_experiment(self) -> Experiment:
        return Experiment(**asdict(self))
