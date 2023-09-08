import pathlib
from dataclasses import dataclass, MISSING
from typing import Optional

from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task
from benchmarl.models.common import ModelConfig
from benchmarl.utils import read_yaml_config


@dataclass
class ExperimentConfig:

    sampling_device: str = MISSING
    train_device: str = MISSING
    gamma: float = MISSING
    polyak_tau: float = MISSING
    lr: float = MISSING
    n_optimizer_steps: int = MISSING
    collected_frames_per_batch: int = MISSING
    n_collection_envs: int = MISSING
    n_iters: int = MISSING
    prefer_continuous_actions: bool = MISSING

    on_policy_minibatch_size: int = MISSING

    off_policy_memory_size: int = MISSING
    off_policy_train_batch_size: int = MISSING
    off_policy_prioritised_alpha: float = MISSING
    off_policy_prioritised_beta: float = MISSING

    def train_batch_size(self, on_policy: bool) -> int:
        return (
            self.collected_frames_per_batch
            if on_policy
            else self.off_policy_train_batch_size
        )

    def train_minibatch_size(self, on_policy: bool) -> int:
        return (
            self.on_policy_minibatch_size
            if on_policy
            else self.train_batch_size(on_policy)
        )

    def replay_buffer_memory_size(self, on_policy: bool) -> int:
        return (
            self.collected_frames_per_batch
            if on_policy
            else self.off_policy_memory_size
        )

    @property
    def traj_len(self) -> int:
        return -(-self.collected_frames_per_batch // self.n_collection_envs)

    @property
    def total_frames(self) -> int:
        return self.n_iters * self.collected_frames_per_batch

    @property
    def exploration_annealing_num_frames(self) -> int:
        return self.total_frames // 2

    @staticmethod
    def get_from_yaml(path: Optional[str] = None):
        if path is None:
            yaml_path = (
                pathlib.Path(__file__).parent
                / "conf"
                / "experiment"
                / "base_experiment.yaml"
            )
            return ExperimentConfig(**read_yaml_config(str(yaml_path.resolve())))
        else:
            return ExperimentConfig(**read_yaml_config(path))


class Experiment:
    def __init__(
        self,
        task: Task,
        algorithm_config: AlgorithmConfig,
        model_config: ModelConfig,
        seed: int,
        config: ExperimentConfig,
    ):
        self.config = config

        self.task = task
        self.model_config = model_config
        self.algorithm_config = algorithm_config
        self.seed = seed

        self.n_iters_performed = 0

        self._setup()

    @property
    def on_policy(self) -> bool:
        return self.algorithm_config.on_policy()

    def _setup(self):
        self._set_action_type()
        self._setup_task()
        self._setup_algorithm()
        self._setup_collector()

    def _set_action_type(self):
        if (
            self.task.supports_continuous_actions()
            and self.algorithm_config.supports_continuous_actions()
            and self.config.prefer_continuous_actions
        ):
            self.continuous_actions = True
        elif (
            self.task.supports_discrete_actions()
            and self.algorithm_config.supports_discrete_actions()
        ):
            self.continuous_actions = False
        elif (
            self.task.supports_continuous_actions()
            and self.algorithm_config.supports_continuous_actions()
        ):
            self.continuous_actions = True
        else:
            raise ValueError(
                f"Algorithm {self.algorithm_config} is not compatible"
                f" with the action space of task {self.task} "
            )

    def _setup_task(self):
        self.env = self.task.get_env(
            num_envs=self.config.n_collection_envs,
            continuous_actions=self.continuous_actions,
            seed=self.seed,
        )
        self.observation_spec = self.task.observation_spec(self.env)
        self.info_spec = self.task.info_spec(self.env)
        self.state_spec = self.task.state_spec(self.env)
        self.action_mask_spec = self.task.action_mask_spec(self.env)
        self.action_spec = self.task.action_spec(self.env)
        self.group_map = self.task.group_map(self.env)

    def _setup_algorithm(self):
        self.algorithm = self.algorithm_config.get_algorithm(
            experiment_config=self.config,
            model_config=self.model_config,
            observation_spec=self.observation_spec,
            action_spec=self.action_spec,
            state_spec=self.state_spec,
            action_mask_spec=self.action_mask_spec,
            group_map=self.group_map,
        )
        self.replay_buffers = {
            group: self.algorithm.get_replay_buffer(
                group=group,
            )
            for group in self.group_map.keys()
        }
        self.losses = {
            group: self.algorithm.get_loss_and_updater(group)[0]
            for group in self.group_map.keys()
        }
        self.target_updaters = {
            group: self.algorithm.get_loss_and_updater(group)[1]
            for group in self.group_map.keys()
        }
        self.optimizers = {
            group: self.algorithm.get_optimizers(group)
            for group in self.group_map.keys()
        }

    def _setup_collector(self):
        self.policy = self.algorithm.get_policy_for_collection()

        self.group_policies = {}
        for group in self.group_map.keys():
            group_policy = self.policy.select_subsequence(out_keys=[(group, "action")])
            assert len(group_policy) == 1
            self.group_policies.update({group: group_policy[0]})

        self.collector = SyncDataCollector(
            self.env,
            self.policy,
            device=self.config.sampling_device,
            storing_device=self.config.train_device,
            frames_per_batch=self.config.collected_frames_per_batch,
            total_frames=self.config.total_frames,
        )

    def run(self):

        for i, batch in enumerate(self.collector):
            assert i == self.n_iters_performed
            print(f"Iteration {i}")
            current_frames = batch.numel()
            for group in self.group_map.keys():
                group_batch = batch.exclude(
                    *[
                        group_name
                        for group_name in self.group_map.keys()
                        if group_name != group
                    ]
                )
                group_batch = self.algorithm.process_batch(group, group_batch)
                group_batch = group_batch.reshape(-1)
                self.replay_buffers[group].extend(group_batch)

                for _ in range(self.config.n_optimizer_steps):
                    for _ in range(
                        self.config.train_batch_size(self.on_policy)
                        // self.config.train_minibatch_size(self.on_policy)
                    ):
                        subdata = self.replay_buffers[group].sample()
                        loss_vals = self.losses[group](subdata)
                        loss_vals = self.algorithm.process_loss_vals(group, loss_vals)
                        for loss_name, loss_value in loss_vals.items():
                            if loss_name in self.optimizers[group].keys():
                                loss_value.backward()
                                self.optimizers[group][loss_name].step()
                                self.optimizers[group][loss_name].zero_grad()
                            elif loss_name.startswith("loss"):
                                assert False
                        if self.target_updaters[group] is not None:
                            self.target_updaters[group].step()

                if isinstance(self.group_policies[group], TensorDictSequential):
                    explore_layer = self.group_policies[group][-1]
                else:
                    explore_layer = self.group_policies[group]
                if hasattr(explore_layer, "step"):  # Step exploration annealing
                    explore_layer.step(current_frames)

            self.collector.update_policy_weights_()

            self.n_iters_performed += 1
            if self.n_iters_performed >= self.config.n_iters:
                break
        self.collector.shutdown()
