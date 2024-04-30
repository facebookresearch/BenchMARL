#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


from typing import Callable, Dict, List, Optional

import numpy as np

from benchmarl.environments.common import Task

from benchmarl.utils import DEVICE_TYPING

from gymnasium import spaces
from pettingzoo import ParallelEnv

from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase, PettingZooWrapper


class MyCustomEnv2(ParallelEnv):
    """
    Multi-agent version of my single agent class.
    """

    metadata = {"render_modes": ["human"], "name": "myclass_v0"}

    def __init__(self, num_envs=2):
        super(MyCustomEnv2, self).__init__()
        self.t = 1
        num_agents = 3
        self.possible_agents = ["player_" + str(r + 1) for r in range(num_agents)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = None

    def observation_space(self, agent):
        state_low = np.concatenate(
            (
                np.zeros(1),  # weights
                np.full(2, -np.inf),
            )
        )
        state_high = np.concatenate(
            (
                np.ones(1),
                np.full(2, np.inf),
            )
        )

        return spaces.Box(
            low=state_low,
            high=state_high,
            shape=(3,),
            dtype=np.float32,  # this was the problem (originally we used float64 but
            # changed to benchmarl error - also the function below)
        )

    def action_space(self, agent):
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        self.t = 1
        self.agents = self.possible_agents[:]
        state_dummy = np.concatenate(
            (
                np.ones(1),
                np.full(2, np.inf),
            )
        )
        observations = {agent: state_dummy for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = observations
        print("RESET DONE")

        return observations, infos

    def step(self, actions):

        self.t += 1

        env_truncation = self.t >= 5

        print(f"step, t: {self.t}")
        print(f"env_truncation: {env_truncation}")
        print()
        self.done = env_truncation
        if not actions:
            self.agents = []

            return {}, {}, {}, {}, {}

        rewards = {}
        observations = {}

        for agent in self.agents:

            state_dummy = np.concatenate(
                (
                    np.ones(1),
                    np.full(2, np.inf),
                )
            )
            reward = 10

            # Store the reward in the dictionary
            observations[agent] = state_dummy
            rewards[agent] = reward

        # self.state = observations
        terminations = {agent: env_truncation for agent in self.agents}
        truncations = {agent: env_truncation for agent in self.agents}

        self.state = observations

        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos


class MyenvTask(Task):

    MY_TASK = None

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:

        return lambda: PettingZooWrapper(
            MyCustomEnv2(),
            categorical_actions=True,
            device=device,
            seed=seed,
            return_state=False,
            **self.config,
        )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_state(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return 100

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return "myenv"
