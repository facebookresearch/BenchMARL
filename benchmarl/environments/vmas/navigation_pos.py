#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING

import torch
from vmas import render_interactively
from vmas.simulator.core import Agent


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    collisions: bool = MISSING
    agents_with_same_goal: int = MISSING
    observe_all_goals: bool = MISSING
    shared_rew: bool = MISSING
    split_goals: bool = MISSING
    lidar_range: float = MISSING
    agent_radius: float = MISSING


from vmas.scenarios.navigation import Scenario


def observation(self, agent: Agent):
    goal_poses = []
    if self.observe_all_goals:
        for a in self.world.agents:
            goal_poses.append(agent.state.pos - a.goal.state.pos)
    else:
        goal_poses.append(agent.state.pos - agent.goal.state.pos)

    return {
        "obs": torch.cat(
            goal_poses
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.collisions
                else []
            ),
            dim=-1,
        ),
        "pos": agent.state.pos,
        "vel": agent.state.vel,
    }


Scenario.observation = observation
NavigationScenario = Scenario


if __name__ == "__main__":
    render_interactively(
        NavigationScenario(),
        control_two_agents=True,
    )
