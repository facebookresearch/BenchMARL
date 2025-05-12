#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import json
import os
import warnings
from collections.abc import MutableMapping, Sequence
from pathlib import Path

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchrl

from tensordict import TensorDictBase
from torch import Tensor

from torchrl.record import TensorboardLogger
from torchrl.record.loggers import get_logger
from torchrl.record.loggers.wandb import WandbLogger

from benchmarl.environments import Task


class Logger:
    def __init__(
        self,
        experiment_name: str,
        folder_name: str,
        experiment_config,
        algorithm_name: str,
        environment_name: str,
        task_name: str,
        model_name: str,
        group_map: Dict[str, List[str]],
        seed: int,
        project_name: str,
        wandb_extra_kwargs: Dict[str, Any],
    ):
        self.experiment_config = experiment_config
        self.algorithm_name = algorithm_name
        self.environment_name = environment_name
        self.task_name = task_name
        self.model_name = model_name
        self.group_map = group_map
        self.seed = seed

        if experiment_config.create_json:
            self.json_writer = JsonWriter(
                folder=folder_name,
                name=experiment_name + ".json",
                algorithm_name=algorithm_name,
                task_name=task_name,
                environment_name=environment_name,
                seed=seed,
            )
        else:
            self.json_writer = None

        self.loggers: List[torchrl.record.loggers.Logger] = []
        for logger_name in experiment_config.loggers:
            wandb_project = wandb_extra_kwargs.get("project", project_name)
            if wandb_project != project_name:
                raise ValueError(
                    f"wandb_extra_kwargs.project ({wandb_project}) is different from the project_name ({project_name})"
                )
            self.loggers.append(
                get_logger(
                    logger_type=logger_name,
                    logger_name=folder_name,
                    experiment_name=experiment_name,
                    wandb_kwargs={
                        "group": task_name,
                        "id": experiment_name,
                        "project": project_name,
                        **wandb_extra_kwargs,
                    },
                )
            )

    def log_hparams(self, **kwargs):
        kwargs.update(
            {
                "algorithm_name": self.algorithm_name,
                "model_name": self.model_name,
                "task_name": self.task_name,
                "environment_name": self.environment_name,
                "seed": self.seed,
            }
        )
        for logger in self.loggers:
            if isinstance(logger, TensorboardLogger):
                # Tensorboard does not like nested dictionaries -> flatten them
                def flatten(dictionary, parent_key="", separator="_"):
                    items = []
                    for key, value in dictionary.items():
                        new_key = parent_key + separator + key if parent_key else key
                        if isinstance(value, MutableMapping):
                            items.extend(
                                flatten(value, new_key, separator=separator).items()
                            )
                        elif isinstance(value, Sequence):
                            for i, v in enumerate(value):
                                items.append((new_key + separator + str(i), v))
                        else:
                            items.append((new_key, value))
                    return dict(items)

                # Convert any non-supported values
                for key, value in kwargs.items():
                    if not isinstance(value, (int, float, str, Tensor)):
                        kwargs[key] = str(value)

                logger.log_hparams(flatten(kwargs))
            else:
                logger.log_hparams(kwargs)

    def log_collection(
        self,
        batch: TensorDictBase,
        task: Task,
        total_frames: int,
        step: int,
    ) -> float:
        to_log = {}
        groups_episode_rewards = []
        gobal_done = self._get_global_done(batch)  # Does not have agent dim
        any_episode_ended = gobal_done.nonzero().numel() > 0
        if not any_episode_ended:
            warnings.warn(
                "No episode terminated this iteration and thus the episode rewards will be NaN, "
                "this is normal if your horizon is longer then one iteration. Learning is proceeding fine."
                "The episodes will probably terminate in a future iteration."
            )
        for group in self.group_map.keys():
            group_episode_rewards = self._log_individual_and_group_rewards(
                group,
                batch,
                gobal_done,
                any_episode_ended,
                to_log,
                log_individual_agents=False,  # Turn on if you want single agent granularity
            )
            # group_episode_rewards has shape (n_episodes) as we took the mean over agents in the group
            groups_episode_rewards.append(group_episode_rewards)

            if "info" in batch.get(("next", group)).keys():
                to_log.update(
                    {
                        f"collection/{group}/info/{key}": value.to(torch.float)
                        .mean()
                        .item()
                        for key, value in batch.get(("next", group, "info")).items()
                    }
                )
        if "info" in batch.keys():
            to_log.update(
                {
                    f"collection/info/{key}": value.to(torch.float).mean().item()
                    for key, value in batch.get(("next", "info")).items()
                }
            )
        to_log.update(task.log_info(batch))
        # global_episode_rewards has shape (n_episodes) as we took the mean over groups
        global_episode_rewards = self._log_global_episode_reward(
            groups_episode_rewards, to_log, prefix="collection"
        )

        self.log(to_log, step=step)
        return global_episode_rewards.mean().item()

    def log_training(self, group: str, training_td: TensorDictBase, step: int):
        if not len(self.loggers):
            return
        to_log = {
            f"train/{group}/{key}": value.mean().item()
            for key, value in training_td.items()
        }
        self.log(to_log, step=step)

    def log_evaluation(
        self,
        rollouts: List[TensorDictBase],
        total_frames: int,
        step: int,
        video_frames: Optional[List] = None,
    ):
        if (
            not len(self.loggers) and not self.experiment_config.create_json
        ) or not len(rollouts):
            return

        # Cut rollouts at first done
        max_length_rollout_0 = 0
        for i in range(len(rollouts)):
            r = rollouts[i]
            next_done = self._get_global_done(r).squeeze(-1)

            # First done index for this traj
            done_index = next_done.nonzero(as_tuple=True)[0]
            if done_index.numel() > 0:
                done_index = done_index[0]
                r = r[: done_index + 1]
            if i == 0:
                max_length_rollout_0 = max(r.batch_size[0], max_length_rollout_0)
            rollouts[i] = r

        to_log = {}
        json_metrics = {}
        for group in self.group_map.keys():
            # returns has shape (n_episodes)
            returns = torch.stack(
                [self._get_reward(group, td).sum(0).mean() for td in rollouts],
                dim=0,
            )
            self._log_min_mean_max(
                to_log, f"eval/{group}/reward/episode_reward", returns
            )
            json_metrics[group + "_return"] = returns

        mean_group_return = self._log_global_episode_reward(
            list(json_metrics.values()), to_log, prefix="eval"
        )
        # mean_group_return has shape (n_episodes) as we take the mean groups
        json_metrics["return"] = mean_group_return

        to_log["eval/reward/episode_len_mean"] = sum(
            td.batch_size[0] for td in rollouts
        ) / len(rollouts)

        if self.json_writer is not None:
            self.json_writer.write(
                metrics=json_metrics,
                total_frames=total_frames,
                evaluation_step=total_frames
                // self.experiment_config.evaluation_interval,
            )
            json_file = str(self.json_writer.path)
            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.save(
                        json_file, base_path=os.path.dirname(json_file)
                    )

        self.log(to_log, step=step)
        if video_frames is not None and max_length_rollout_0 > 1:
            video_frames = np.stack(video_frames[: max_length_rollout_0 - 1], axis=0)
            vid = torch.tensor(
                np.transpose(video_frames, (0, 3, 1, 2)),
                dtype=torch.uint8,
            ).unsqueeze(0)
            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_video("eval/video", vid, fps=20, commit=False)
                else:
                    # Other loggers cannot deal with odd video sizes so we check if the video dimensions are odd and make them even
                    for index in (-1, -2):
                        if vid.shape[index] % 2 != 0:
                            vid = vid.index_select(
                                index, torch.arange(1, vid.shape[index])
                            )
                    # End of check

                    logger.log_video("eval_video", vid, step=step)

    def commit(self):
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log({}, commit=True)

    def log(self, dict_to_log: Dict, step: int = None):
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log(dict_to_log, commit=False)
            else:
                for key, value in dict_to_log.items():
                    logger.log_scalar(key.replace("/", "_"), value, step=step)

    def finish(self):
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                wandb.finish()

    def _get_reward(
        self, group: str, td: TensorDictBase, remove_agent_dim: bool = False
    ):
        reward = td.get(("next", group, "reward"), None)
        if reward is None:
            reward = (
                td.get(("next", "reward")).expand(td.get(group).shape).unsqueeze(-1)
            )
        return reward.mean(-2) if remove_agent_dim else reward

    def _get_agents_done(
        self, group: str, td: TensorDictBase, remove_agent_dim: bool = False
    ):
        done = td.get(("next", group, "done"), None)
        if done is None:
            done = td.get(("next", "done")).expand(td.get(group).shape).unsqueeze(-1)

        return done.any(-2) if remove_agent_dim else done

    def _get_global_done(
        self,
        td: TensorDictBase,
    ):
        done = td.get(("next", "done"))
        return done

    def _get_episode_reward(
        self, group: str, td: TensorDictBase, remove_agent_dim: bool = False
    ):
        episode_reward = td.get(("next", group, "episode_reward"), None)
        if episode_reward is None:
            episode_reward = (
                td.get(("next", "episode_reward"))
                .expand(td.get(group).shape)
                .unsqueeze(-1)
            )
        return episode_reward.mean(-2) if remove_agent_dim else episode_reward

    def _log_individual_and_group_rewards(
        self,
        group: str,
        batch: TensorDictBase,
        global_done: Tensor,
        any_episode_ended: bool,
        to_log: Dict[str, Tensor],
        prefix: str = "collection",
        log_individual_agents: bool = True,
    ):
        reward = self._get_reward(group, batch)  # Has agent dim
        episode_reward = self._get_episode_reward(group, batch)  # Has agent dim
        n_agents_in_group = episode_reward.shape[-2]

        # Add multiagent dim
        unsqueeze_global_done = global_done.unsqueeze(-1).expand(
            (*batch.get_item_shape(group), 1)
        )
        #######
        # All trajectories are considered done at the global done
        #######

        # 1. Here we log rewards from individual agent data
        if log_individual_agents:
            for i in range(n_agents_in_group):
                self._log_min_mean_max(
                    to_log,
                    f"{prefix}/{group}/reward/agent_{i}/reward",
                    reward[..., i, :],
                )
                if any_episode_ended:
                    agent_global_done = unsqueeze_global_done[..., i, :]
                    self._log_min_mean_max(
                        to_log,
                        f"{prefix}/{group}/reward/agent_{i}/episode_reward",
                        episode_reward[..., i, :][agent_global_done],
                    )

        # 2. Here we log rewards from group data taking the mean over agents
        group_episode_reward = episode_reward.mean(-2)[global_done]
        if any_episode_ended:
            self._log_min_mean_max(
                to_log, f"{prefix}/{group}/reward/episode_reward", group_episode_reward
            )
        self._log_min_mean_max(to_log, f"{prefix}/reward/reward", reward)

        return group_episode_reward

    def _log_global_episode_reward(
        self, episode_rewards: List[Tensor], to_log: Dict[str, Tensor], prefix: str
    ):
        # Each element in the list is the episode reward (with shape n_episodes) for the group at the global done,
        # so they will have same shape as done is shared
        episode_rewards = torch.stack(episode_rewards, dim=0).mean(
            0
        )  # Mean over groups
        if episode_rewards.numel() > 0:
            self._log_min_mean_max(
                to_log, f"{prefix}/reward/episode_reward", episode_rewards
            )

        return episode_rewards

    def _log_min_mean_max(self, to_log: Dict[str, Tensor], key: str, value: Tensor):
        to_log.update(
            {
                key + "_min": value.min().item(),
                key + "_mean": value.mean().item(),
                key + "_max": value.max().item(),
            }
        )


class JsonWriter:
    """
    Writer to create json files for reporting according to marl-eval

    Follows conventions from https://github.com/instadeepai/marl-eval/tree/main#usage-

    Args:
        folder (str): folder where to write the file
        name (str): file name
        algorithm_name (str): algorithm name
        task_name (str): task name
        environment_name (str): environment name
        seed (int): seed of the experiment

    """

    def __init__(
        self,
        folder: str,
        name: str,
        algorithm_name: str,
        task_name: str,
        environment_name: str,
        seed: int,
    ):
        self.path = Path(folder) / Path(name)
        self.run_data = {"absolute_metrics": {}}
        self.data = {
            environment_name: {
                task_name: {algorithm_name: {f"seed_{seed}": self.run_data}}
            }
        }

    def write(
        self, total_frames: int, metrics: Dict[str, List[Tensor]], evaluation_step: int
    ):
        """
        Writes a step into the json reporting file

        Args:
            total_frames (int): total frames collected so far in the experiment
            metrics (dictionary mapping str to tensor): each value is a 1-dim tensor for the metric in key
                of len equal to the number of evaluation episodes for this step.
            evaluation_step (int): the evaluation step

        """
        metrics = {k: val.tolist() for k, val in metrics.items()}
        step_metrics = {"step_count": total_frames}
        step_metrics.update(metrics)
        step_str = f"step_{evaluation_step}"
        if step_str in self.run_data:
            self.run_data[step_str].update(step_metrics)
        else:
            self.run_data[step_str] = step_metrics

        # Store the maximum of each metric
        for metric_name in metrics.keys():
            if len(metrics[metric_name]):
                max_metric = max(metrics[metric_name])
                if metric_name in self.run_data["absolute_metrics"]:
                    prev_max_metric = self.run_data["absolute_metrics"][metric_name][0]
                    max_metric = max(max_metric, prev_max_metric)
                self.run_data["absolute_metrics"][metric_name] = [max_metric]

        with open(self.path, "w+") as f:
            json.dump(self.data, f, indent=4)
