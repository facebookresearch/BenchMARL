import os
from pathlib import Path

import torch

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig, RmGnnConfig
from tensordict import TensorDict
from torch import nn
from torchrl.envs.utils import ExplorationType, set_exploration_type


def get_policy():
    current_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    config_folder = current_folder / "yaml"

    config = ExperimentConfig.get_from_yaml(str(config_folder / "experiment.yaml"))
    config.restore_file = str(current_folder / "checkpoint.pt")

    experiment = Experiment(
        task=VmasTask.RM_NAVIGATION.get_from_yaml(
            str(config_folder / "rm_navigation.yaml")
        ),
        algorithm_config=MappoConfig.get_from_yaml(str(config_folder / "mappo.yaml")),
        model_config=RmGnnConfig.get_from_yaml(str(config_folder / "rmgnn.yaml")),
        config=ExperimentConfig.get_from_yaml(str(config_folder / "experiment.yaml")),
        critic_model_config=MlpConfig(
            num_cells=[256, 256], activation_class=nn.Tanh, layer_class=nn.Linear
        ),
        seed=0,
    )

    return experiment.policy


def run_policy(policy):
    n_agents = 2

    # These are he input args
    pos = torch.rand((1, n_agents, 2), dtype=torch.float)
    vel = torch.rand((1, n_agents, 2), dtype=torch.float)
    goal = torch.rand((1, n_agents, 2), dtype=torch.float)

    rel_goal_pos = pos - goal

    obs = torch.cat([pos, vel, rel_goal_pos], dim=-1)
    td = TensorDict(
        {"agents": TensorDict({"observation": obs}, batch_size=[1, n_agents])},
        batch_size=[1],
    )
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        out_td = policy(td)

    return out_td.get(("agents", "action"))  # shape 1 (should squeeze), n_agents, 2


if __name__ == "__main__":
    policy = get_policy()
    run_policy(policy)
