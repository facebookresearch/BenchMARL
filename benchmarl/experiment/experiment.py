import pathlib
import time
from dataclasses import dataclass, MISSING
from typing import List, Optional

import torch

from tensordict import TensorDictBase
from tensordict.nn import TensorDictSequential
from tensordict.utils import _unravel_key_to_tuple
from torchrl.collectors import SyncDataCollector
from torchrl.envs import EnvBase, RewardSum, SerialEnv, TransformedEnv
from torchrl.envs.transforms import Compose
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task
from benchmarl.experiment.logger import MultiAgentLogger
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
    n_envs_per_worker: int = MISSING
    n_iters: int = MISSING
    prefer_continuous_actions: bool = MISSING
    clip_grad_norm: bool = MISSING
    clip_grad_val: Optional[float] = MISSING

    on_policy_minibatch_size: int = MISSING

    off_policy_memory_size: int = MISSING
    off_policy_train_batch_size: int = MISSING
    off_policy_prioritised_alpha: float = MISSING
    off_policy_prioritised_beta: float = MISSING

    evaluation: bool = MISSING
    evaluation_interval: int = MISSING
    evaluation_episodes: int = MISSING

    loggers: List[str] = MISSING
    create_json: bool = MISSING

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
        return -(-self.collected_frames_per_batch // self.n_envs_per_worker)

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
                pathlib.Path(__file__).parent.parent
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

        self._setup()

    @property
    def on_policy(self) -> bool:
        return self.algorithm_config.on_policy()

    def _setup(self):
        self._set_action_type()
        self._setup_task()
        self._setup_algorithm()
        self._setup_collector()
        self._setup_logger()

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
        test_env = self.model_config.process_env_fun(
            self.task.get_env_fun(
                num_envs=self.config.evaluation_episodes,
                continuous_actions=self.continuous_actions,
                seed=self.seed,
            )
        )()
        env_func = self.model_config.process_env_fun(
            self.task.get_env_fun(
                num_envs=self.config.n_envs_per_worker,
                continuous_actions=self.continuous_actions,
                seed=self.seed,
            )
        )

        self.observation_spec = self.task.observation_spec(test_env)
        self.info_spec = self.task.info_spec(test_env)
        self.state_spec = self.task.state_spec(test_env)
        self.action_mask_spec = self.task.action_mask_spec(test_env)
        self.action_spec = self.task.action_spec(test_env)
        self.group_map = self.task.group_map(test_env)

        reward_spec = test_env.output_spec["full_reward_spec"]
        transforms = []
        for reward_key in reward_spec.keys(True, True):
            reward_key = _unravel_key_to_tuple(reward_key)
            transforms.append(
                RewardSum(
                    in_keys=[reward_key],
                    out_keys=[reward_key[:-1] + ("episode_reward",)],
                )
            )
        transform = Compose(*transforms)

        def env_func_transformed() -> EnvBase:
            return TransformedEnv(env_func(), transform)

        if test_env.batch_size == ():
            self.env_func = lambda: SerialEnv(
                self.config.evaluation_episodes, env_func_transformed
            )
            self.test_env = SerialEnv(self.config.evaluation_episodes, lambda: test_env)
        else:
            self.env_func = env_func_transformed
            self.test_env = test_env

        assert self.test_env.batch_size == (self.config.evaluation_episodes,)

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
            self.env_func,
            self.policy,
            device=self.config.sampling_device,
            storing_device=self.config.train_device,
            frames_per_batch=self.config.collected_frames_per_batch,
            total_frames=self.config.total_frames,
        )

    def _setup_logger(self):
        self.logger = MultiAgentLogger(
            self.config,
            algorithm_name=self.algorithm_config.associated_class().__name__.lower(),
            model_name=self.model_config.associated_class().__name__.lower(),
            environment_name=self.task.env_name().lower(),
            task_name=self.task.name.lower(),
            group_map=self.group_map,
            seed=self.seed,
        )
        self.logger.log_hparams(
            experiment_config=self.config.__dict__,
            algorithm_config=self.algorithm_config.__dict__,
            model_config=self.model_config.__dict__,
            task_config=self.task.config,
            continuous_actions=self.continuous_actions,
            on_policy=self.on_policy,
        )

    def run(self):
        self._collection_loop()

    def _collection_loop(self):
        self.total_time = 0
        self.total_frames = 0
        self.n_iters_performed = 0
        sampling_start = time.time()

        # Training/collection iterations
        for batch in self.collector:
            print(f"Iteration {self.n_iters_performed}")

            # Logging collection
            collection_time = time.time() - sampling_start
            current_frames = batch.numel()
            self.total_frames += current_frames
            self.logger.log_collection(
                batch, self.total_frames, step=self.n_iters_performed
            )

            # Loop over groups
            training_start = time.time()
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

                training_tds = []
                for _ in range(self.config.n_optimizer_steps):
                    for _ in range(
                        self.config.train_batch_size(self.on_policy)
                        // self.config.train_minibatch_size(self.on_policy)
                    ):
                        training_tds.append(self._optimizer_loop(group))

                self.logger.log_training(
                    group, torch.stack(training_tds), step=self.n_iters_performed
                )

                # Exploration update
                if isinstance(self.group_policies[group], TensorDictSequential):
                    explore_layer = self.group_policies[group][-1]
                else:
                    explore_layer = self.group_policies[group]
                if hasattr(explore_layer, "step"):  # Step exploration annealing
                    explore_layer.step(current_frames)

            # Update policy in collector
            self.collector.update_policy_weights_()

            # Timers
            training_time = time.time() - training_start
            iteration_time = collection_time + training_time
            self.total_time += iteration_time
            self.logger.log(
                {
                    "timers/collection_time": collection_time,
                    "timers/training_time": training_time,
                    "timers/iteration_time": iteration_time,
                    "timers/total_time": self.total_time,
                    "counters/current_frames": current_frames,
                    "counters/total_frames": self.total_frames,
                    "counters/total_iter": self.n_iters_performed,
                },
                step=self.n_iters_performed,
            )

            # Evaluation
            if (
                self.config.evaluation_episodes > 0
                and self.n_iters_performed % self.config.evaluation_interval == 0
            ):
                self._evaluation_loop(iter=self.n_iters_performed)

            self.n_iters_performed += 1
            self.logger.commit()
            sampling_start = time.time()

        self.collector.shutdown()

    def _optimizer_loop(self, group: str) -> TensorDictBase:
        subdata = self.replay_buffers[group].sample()
        loss_vals = self.losses[group](subdata)
        training_td = loss_vals.detach()
        loss_vals = self.algorithm.process_loss_vals(group, loss_vals)

        for loss_name, loss_value in loss_vals.items():
            if loss_name in self.optimizers[group].keys():
                optimizer = self.optimizers[group][loss_name]

                loss_value.backward()

                grad_norm = self._grad_clip(optimizer)
                training_td.set(
                    f"grad_norm_{loss_name}",
                    torch.tensor(grad_norm, device=self.config.train_device),
                )

                optimizer.step()
                optimizer.zero_grad()
            elif loss_name.startswith("loss"):
                assert False
        if self.target_updaters[group] is not None:
            self.target_updaters[group].step()
        return training_td

    def _grad_clip(self, optimizer: torch.optim.Optimizer) -> float:
        params = []
        for param_group in optimizer.param_groups:
            params += param_group["params"]

        if self.config.clip_grad_norm and self.config.clip_grad_val is not None:
            gn = torch.nn.utils.clip_grad_norm_(params, self.config.clip_grad_val)
        else:
            gn = sum(p.grad.pow(2).sum() for p in params if p.grad is not None).sqrt()
            if self.config.clip_grad_val is not None:
                torch.nn.utils.clip_grad_value_(params, self.config.clip_grad_val)

        return float(gn)

    def _evaluation_loop(self, iter: int):
        evaluation_start = time.time()
        with torch.no_grad() and set_exploration_type(ExplorationType.MODE):
            if self.task.has_render():
                frames = []

                def callback(env, td):
                    frames.append(env.render(mode="rgb_array"))

            else:
                frames = None
                callback = None

            rollouts = self.test_env.rollout(
                max_steps=self.task.max_steps(),
                policy=self.policy,
                callback=callback,
                auto_cast_to_device=True,
                break_when_any_done=False,
                # We are running vectorized evaluation we do not want it to stop when just one env is done
            )
        evaluation_time = time.time() - evaluation_start
        self.logger.log({"timers/evaluation_time": evaluation_time}, step=iter)
        self.logger.log_evaluation(rollouts, frames, step=iter)