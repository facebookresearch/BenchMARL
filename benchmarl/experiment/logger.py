#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchrl

from tensordict import TensorDictBase
from torch import Tensor
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
            self.loggers.append(
                get_logger(
                    logger_type=logger_name,
                    logger_name=folder_name,
                    experiment_name=experiment_name,
                    wandb_kwargs={
                        "group": task_name,
                        "project": "benchmarl",
                        "id": experiment_name,
                    },
                )
            )

    def log_hparams(self, **kwargs):
        for logger in self.loggers:
            kwargs.update(
                {
                    "algorithm_name": self.algorithm_name,
                    "model_name": self.model_name,
                    "task_name": self.task_name,
                    "environment_name": self.environment_name,
                    "seed": self.seed,
                }
            )
            logger.log_hparams(kwargs)

    def log_collection(
        self,
        batch: TensorDictBase,
        task: Task,
        total_frames: int,
        step: int,
    ) -> float:

        to_log = {}
        json_metrics = {}
        for group in self.group_map.keys():
            episode_reward = self._get_episode_reward(group, batch)
            done = self._get_done(group, batch)
            reward = self._get_reward(group, batch)
            to_log.update(
                {
                    f"collection/{group}/reward/reward_min": reward.min().item(),
                    f"collection/{group}/reward/reward_mean": reward.mean().item(),
                    f"collection/{group}/reward/reward_max": reward.max().item(),
                }
            )
            json_metrics[group + "_return"] = episode_reward.mean(-2)[done.any(-2)]
            episode_reward = episode_reward[done]
            if episode_reward.numel() > 0:
                to_log.update(
                    {
                        f"collection/{group}/reward/episode_reward_min": episode_reward.min().item(),
                        f"collection/{group}/reward/episode_reward_mean": episode_reward.mean().item(),
                        f"collection/{group}/reward/episode_reward_max": episode_reward.max().item(),
                    }
                )
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
        mean_group_return = torch.stack(
            [value for key, value in json_metrics.items()], dim=0
        ).mean(0)
        if mean_group_return.numel() > 0:
            to_log.update(
                {
                    "collection/reward/episode_reward_min": mean_group_return.min().item(),
                    "collection/reward/episode_reward_mean": mean_group_return.mean().item(),
                    "collection/reward/episode_reward_max": mean_group_return.max().item(),
                }
            )
        self.log(to_log, step=step)
        return mean_group_return.mean().item()

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
        to_log = {}
        json_metrics = {}
        for group in self.group_map.keys():
            # Cut the rollouts at the first done
            for k, r in enumerate(rollouts):
                next_done = self._get_done(group, r)
                # Reduce it to batch size
                next_done = next_done.sum(
                    tuple(range(r.batch_dims, next_done.ndim)),
                    dtype=torch.bool,
                )
                # First done index for this traj
                done_index = next_done.nonzero(as_tuple=True)[0]
                if done_index.numel() > 0:
                    done_index = done_index[0]
                    rollouts[k] = r[: done_index + 1]

            returns = [
                self._get_reward(group, td).sum(0).mean().item() for td in rollouts
            ]
            json_metrics[group + "_return"] = torch.tensor(
                returns, device=rollouts[0].device
            )
            to_log.update(
                {
                    f"eval/{group}/reward/episode_reward_min": min(returns),
                    f"eval/{group}/reward/episode_reward_mean": sum(returns)
                    / len(rollouts),
                    f"eval/{group}/reward/episode_reward_max": max(returns),
                }
            )

        mean_group_return = torch.stack(
            [value for key, value in json_metrics.items()], dim=0
        ).mean(0)
        to_log.update(
            {
                "eval/reward/episode_reward_min": mean_group_return.min().item(),
                "eval/reward/episode_reward_mean": mean_group_return.mean().item(),
                "eval/reward/episode_reward_max": mean_group_return.max().item(),
                "eval/reward/episode_len_mean": sum(td.batch_size[0] for td in rollouts)
                / len(rollouts),
            }
        )
        json_metrics["return"] = mean_group_return
        if self.json_writer is not None:
            self.json_writer.write(
                metrics=json_metrics,
                total_frames=total_frames,
                evaluation_step=total_frames
                // self.experiment_config.evaluation_interval,
            )
        self.log(to_log, step=step)
        if video_frames is not None:
            vid = torch.tensor(
                np.transpose(video_frames[: rollouts[0].batch_size[0]], (0, 3, 1, 2)),
                dtype=torch.uint8,
            ).unsqueeze(0)
            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_video("eval/video", vid, fps=20, commit=False)
                else:
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
        if ("next", group, "reward") not in td.keys(True, True):
            reward = (
                td.get(("next", "reward")).expand(td.get(group).shape).unsqueeze(-1)
            )
        else:
            reward = td.get(("next", group, "reward"))
        return reward.mean(-2) if remove_agent_dim else reward

    def _get_done(self, group: str, td: TensorDictBase, remove_agent_dim: bool = False):
        if ("next", group, "done") not in td.keys(True, True):
            done = td.get(("next", "done")).expand(td.get(group).shape).unsqueeze(-1)
        else:
            done = td.get(("next", group, "done"))
        return done.any(-2) if remove_agent_dim else done

    def _get_episode_reward(
        self, group: str, td: TensorDictBase, remove_agent_dim: bool = False
    ):
        if ("next", group, "episode_reward") not in td.keys(True, True):
            episode_reward = (
                td.get(("next", "episode_reward"))
                .expand(td.get(group).shape)
                .unsqueeze(-1)
            )
        else:
            episode_reward = td.get(("next", group, "episode_reward"))
        return episode_reward.mean(-2) if remove_agent_dim else episode_reward


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
