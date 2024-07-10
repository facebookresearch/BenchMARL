import os
from pathlib import Path

import torch
import vmas

from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import RmGnnConfig
from tensordict import TensorDict

from torchrl.envs.utils import ExplorationType, set_exploration_type


def get_policy():
    current_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    config_folder = current_folder / "yaml"

    config = ExperimentConfig.get_from_yaml(str(config_folder / "experiment.yaml"))
    config.restore_file = str(current_folder / "checkpoint.pt")

    experiment = Experiment(
        config=config,
        task=VmasTask.RM_NAVIGATION.get_from_yaml(
            str(config_folder / "rm_navigation.yaml")
        ),
        algorithm_config=IppoConfig.get_from_yaml(str(config_folder / "ippo.yaml")),
        model_config=RmGnnConfig.get_from_yaml(str(config_folder / "rmgnn.yaml")),
        critic_model_config=RmGnnConfig.get_from_yaml(
            str(config_folder / "rmgnn.yaml")
        ),
        seed=0,
    )

    return experiment.policy


def run_policy(policy, obs):
    n_agents = 9

    # These are he input args
    # pos = torch.zeros((1, n_agents, 2), dtype=torch.float)
    # vel = torch.zeros((1, n_agents, 2), dtype=torch.float)
    #
    # goal = pos.clone()
    #
    # rel_goal_pos = pos - goal
    #
    # obs = torch.cat([pos, vel, rel_goal_pos], dim=-1)
    td = TensorDict(
        {"agents": TensorDict({"observation": obs}, batch_size=[1, n_agents])},
        batch_size=[1],
    )
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        out_td = policy(td)

    return out_td.get(("agents", "action"))  # shape 1 (should squeeze), n_agents, 2


if __name__ == "__main__":
    n_agents = 9
    policy = get_policy()

    env = vmas.make_env(
        scenario="rm_navigation",
        num_envs=1,
        continuous_actions=True,
        # Environment specific variables
        n_agents=n_agents,
        v_range=2,
        a_range=2,
    )
    obs = torch.stack(env.reset(), dim=-2)
    frame_list = []
    for _ in range(200):

        actions = run_policy(policy, obs).unbind(-2)

        obs, rews, dones, info = env.step(actions)
        obs = torch.stack(obs, dim=-2)
        frame = env.render(
            mode="rgb_array",
            visualize_when_rgb=True,
        )
        frame_list.append(frame)

    # vmas.simulator.utils.save_video(
    #     "rm_navigation", frame_list, fps=1 / env.scenario.world.dt
    # )
