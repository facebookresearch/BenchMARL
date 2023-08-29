from dataclasses import dataclass

from torchrl.collectors import SyncDataCollector

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task
from benchmarl.models.common import ModelConfig
from benchmarl.utils import DEVICE_TYPING


@dataclass
class ExperimentConfig:

    sampling_device: DEVICE_TYPING = "cpu"
    train_device: DEVICE_TYPING = "cpu"

    gamma: float = 0.9
    polyak_tau: float = 0.005

    lr: float = 3e-5
    n_optimizer_steps: int = 10
    collected_frames_per_batch: int = 1000
    n_collection_envs: int = 1
    n_iters: int = 100
    prefer_continuous_actions: bool = True

    on_policy_minibatch_size: int = 100

    off_policy_memory_size: int = 100_000
    off_policy_train_batch_size: int = 10_000
    off_policy_prioritised_alpha: float = 0.7
    off_policy_prioritised_beta: float = 0.5

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
        return self.algorithm_config.associated_class().on_policy()

    def _setup(self):
        self._set_action_type()
        self._setup_task()
        self._setup_algorithm()
        self._setup_collector()

    def _set_action_type(self):
        if (
            self.task.supports_continuous_actions()
            and self.algorithm_config.associated_class().supports_continuous_actions()
            and self.config.prefer_continuous_actions
        ):
            self.continuous_actions = True
        elif (
            self.task.supports_discrete_actions()
            and self.algorithm_config.associated_class().supports_discrete_actions()
        ):
            self.continuous_actions = False
        elif (
            self.task.supports_continuous_actions()
            and self.algorithm_config.associated_class().supports_continuous_actions()
        ):
            self.continuous_actions = True
        else:
            raise ValueError(
                f"Algorithm {self.algorithm_config.associated_class()} is not compatible"
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
            if hasattr(self.policy, "step"):  # Step exploration annealing
                self.policy.step(current_frames)
            self.collector.update_policy_weights_()

            self.n_iters_performed += 1
            if self.n_iters_performed >= self.config.n_iters:
                break
        self.collector.shutdown()