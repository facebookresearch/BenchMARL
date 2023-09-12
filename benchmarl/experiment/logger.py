import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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

        cwd = (
            os.getcwd()
            if not HydraConfig.initialized()
            else HydraConfig.get().runtime.output_dir
        )
        exp_name = generate_exp_name(f"{algorithm_name}_{task_name}_{model_name}", "")

        if experiment_config.create_json:
            self.json_writer = JsonWriter(folder=cwd, name=exp_name + ".json")
        else:
            self.json_writer = None

        self.loggers: List[Logger] = []
        for logger_name in experiment_config.loggers:

            self.loggers.append(
                get_logger(
                    logger_type=logger_name,
                    logger_name=cwd,
                    experiment_name=exp_name,
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
        total_frames: int,
        step: int,
    ):
        if not len(self.loggers) and self.json_writer is None:
            return
        to_log = {}
        group_returns = {}
        for group in self.group_map.keys():
            episode_reward = self._get_episode_reward(group, batch)
            done = self._get_done(group, batch)
            group_returns[group + "_return"] = episode_reward.mean(-2)[
                done.any(-2)
            ].tolist()
            reward = self._get_reward(group, batch)
            episode_reward = episode_reward[done]
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
        if self.json_writer is not None:
            self.json_writer.write(
                metrics=group_returns, total_frames=total_frames, step=step
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
        self.log(to_log, step=step)

        if frames is not None:
            vid = torch.tensor(
                np.transpose(frames[: rollouts[0].batch_size[0]], (0, 3, 1, 2)),
                dtype=torch.uint8,
            ).unsqueeze(0)
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

    def _get_reward(
        self, group: str, td: TensorDictBase, remove_agent_dim: bool = False
    ):
        if ("next", group, "reward") not in td.keys(True, True):
            reward = (
                td.get(("next", "reward")).expand(td.get(group).shape).unsqueeze(-1),
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
                .unsqueeze(-1),
            )
        else:
            episode_reward = td.get(("next", group, "episode_reward"))
        return episode_reward.mean(-2) if remove_agent_dim else episode_reward


class JsonWriter:
    def __init__(self, folder: str, name: str):
        self.path = Path(folder) / Path(name)
        self.data = {}

    def write(self, total_frames: int, metrics: Dict[str, Any], step: int):
        metrics.update({"step_count": total_frames})
        step_str = f"step_{step}"
        if step_str in self.data:
            self.data[step_str].update(metrics)
        else:
            self.data[step_str] = metrics

        with open(self.path, "w+") as f:
            json.dump(self.data, f, indent=4)
