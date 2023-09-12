import os
from typing import Dict, List, Optional

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig

from tensordict import TensorDictBase
from torchrl.record.loggers import generate_exp_name, get_logger, Logger
from torchrl.record.loggers.wandb import WandbLogger


class MultiAgentLogger:
    def __init__(
        self,
        experiment_config,
        algorithm_name: str,
        task_name: str,
        model_name: str,
        group_map: Dict[str, List[str]],
    ):
        self.experiment_config = experiment_config
        self.algorithm_name = algorithm_name
        self.task_name = task_name
        self.model_name = model_name
        self.group_map = group_map

        self.loggers: List[Logger] = []
        for logger_name in experiment_config.loggers:
            cwd = (
                os.getcwd()
                if not HydraConfig.initialized()
                else HydraConfig.get().runtime.output_dir
            )
            self.loggers.append(
                get_logger(
                    logger_type=logger_name,
                    logger_name=cwd,
                    experiment_name=generate_exp_name(
                        f"{algorithm_name}_{task_name}_{model_name}", ""
                    ),
                    wandb_kwargs={
                        "group": task_name,
                        "project": "benchmarl",
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
                }
            )
            logger.log_hparams(kwargs)

    def log_collection(
        self,
        batch: TensorDictBase,
        step: int,
    ):
        if not len(self.loggers):
            return
        to_log = {}
        for group in self.group_map.keys():
            reward = self._get_reward(group, batch)
            episode_reward = self._get_episode_reward(group, batch)

            to_log.update(
                {
                    f"collection/{group}/reward/reward_min": reward.min().item(),
                    f"collection/{group}/reward/reward_mean": reward.mean().item(),
                    f"collection/{group}/reward/reward_max": reward.max().item(),
                    f"collection/{group}/reward/episode_reward_min": episode_reward.min().item(),
                    f"collection/{group}/reward/episode_reward_mean": episode_reward.mean().item(),
                    f"collection/{group}/reward/episode_reward_max": episode_reward.max().item(),
                }
            )
            if "info" in batch.get(group).keys():
                to_log.update(
                    {
                        f"collection/{group}/info/{key}": value.mean().item()
                        for key, value in batch.get((group, "info")).items()
                    }
                )
        self.log(to_log, step=step)

    def log_training(self, group: str, training_td: TensorDictBase, step: int):
        if not len(self.loggers):
            return
        to_log = {
            f"train/{group}/{key}": value.mean().item()
            for key, value in training_td.items()
        }
        self.log(to_log, step=step)

    def log_evaluation(
        self, rollouts=TensorDictBase, frames: Optional[List] = None, step: int = None
    ):
        if not len(self.loggers):
            return
        to_log = {}
        # Unbind vectorized dim
        rollouts = list(rollouts.unbind(dim=0))
        for group in self.group_map.keys():
            for k, r in enumerate(rollouts):
                next_done = self._get_done(group, r)
                # Reduce it to batch size
                next_done = next_done.sum(
                    tuple(range(r.batch_dims, next_done.ndim)),
                    dtype=torch.bool,
                )
                done_index = next_done.nonzero(as_tuple=True)[0][
                    0
                ]  # First done index for this traj
                rollouts[k] = r[: done_index + 1]

            rewards = [self._get_reward(group, td).sum(0).mean() for td in rollouts]
            to_log.update(
                {
                    f"eval/{group}/episode_reward_min": min(rewards),
                    f"eval/{group}/episode_reward_mean": sum(rewards) / len(rollouts),
                    f"eval/{group}/episode_reward_max": max(rewards),
                    f"eval/{group}/episode_len_mean": sum(
                        td.batch_size[0] for td in rollouts
                    )
                    / len(rollouts),
                }
            )

        vid = torch.tensor(
            np.transpose(frames[: rollouts[0].batch_size[0]], (0, 3, 1, 2)),
            dtype=torch.uint8,
        ).unsqueeze(0)
        self.log(to_log, step=step)

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                logger.experiment.log(
                    {
                        "eval/video": wandb.Video(vid, fps=20, format="mp4"),
                    },
                    commit=False,
                )
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

    def _get_reward(self, group: str, td: TensorDictBase):
        if ("next", group, "reward") not in td.keys(True, True):
            reward = (
                td.get(("next", "reward")).expand(td.get(group).shape).unsqueeze(-1),
            )
        else:
            reward = td.get(("next", group, "reward"))
        return reward

    def _get_done(self, group: str, td: TensorDictBase):
        if ("next", group, "done") not in td.keys(True, True):
            done = (td.get(("next", "done")).expand(td.get(group).shape).unsqueeze(-1),)
        else:
            done = td.get(("next", group, "done"))
        return done

    def _get_episode_reward(self, group: str, td: TensorDictBase):
        if ("next", group, "episode_reward") not in td.keys(True, True):
            episode_reward = (
                td.get(("next", "episode_reward"))
                .expand(td.get(group).shape)
                .unsqueeze(-1),
            )
        else:
            episode_reward = td.get(("next", group, "episode_reward"))
        done = self._get_done(group, td)

        return episode_reward[done]
