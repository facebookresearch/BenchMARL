#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDictBase
from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase
#from torchrl.envs.libs import YourTorchRLEnvConstructor 
from .Info_relay_env_wrapper import InfoRelayEnv #a torchrl wrapper for our env

from torchrl.data import Composite


class CustomEnvTask(Task):
    # Your task names.
    # Their config will be loaded from conf/task/customenv

    INFO_RELAY = None  # Loaded automatically from conf/task/customenv/info_relay
    SIMPLE_TEST = None  # Loaded automatically from conf/task/customenv/task_2

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        return lambda: InfoRelayEnv(
            #scenario=self.name.lower(),
            #num_envs=num_envs,  # Number of vectorized envs (do not use this param if the env is not vectorized)
            continuous_actions=continuous_actions,  # Ignore this param if your env does not have this choice
            seed=seed,
            device=device,
            categorical_actions=True,  # If your env has discrete actions, they need to be categorical (TorchRL can help with this)
            render_mode = "rgb_array",
            **self.config,  # Pass the loaded config (this is what is in your yaml
        )

    def supports_continuous_actions(self) -> bool:
        # Does the environment support continuous actions?
        return True

    def supports_discrete_actions(self) -> bool:
        # Does the environment support discrete actions?
        return True

    def has_render(self, env: EnvBase) -> bool:
        # Does the env have a env.render(mode="rgb_array") or env.render() function?
        return True

    def max_steps(self, env: EnvBase) -> int:
        # Maximum number of steps for a rollout during evaluation
        return self.config["max_cycles"] 

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # The group map mapping group names to agent names
        # The data in the tensordict will havebe presented this way
        #return {"agents": [agent for agent in env.agents]}#{"agents": [agent.name for agent in env.agents]} # obs ser ut som att de vill ha 'hela agentobjektet' och inte bara namnet som agenter
        return env.group_map

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the observation.
        # Must be a CompositeSpec with one (group_name, observation_key) entry per group.
        #return env.full_observation_spec.clone()
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the action.
        # If provided, must be a CompositeSpec with one (group_name, "action") entry per group.
        return env.full_action_spec

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the state.
        # If provided, must be a CompositeSpec with one "state" entry
        #return env.observation_spec.clone()
        if "state" in env.observation_spec:
            return Composite({"state": env.observation_spec["state"].clone()})
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the action mask.
        # If provided, must be a CompositeSpec with one (group_name, "action_mask") entry per group.
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "action_mask":
                    del group_obs_spec[key]
            if group_obs_spec.is_empty():
                del observation_spec[group]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        if observation_spec.is_empty():
            return None
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the info.
        # If provided, must be a CompositeSpec with one (group_name, "info") entry per group (this entry can be composite).
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        return observation_spec

    @staticmethod
    def env_name() -> str:
        # The name of the environment in the benchmarl/conf/task folder
        return "customenv"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        # Optionally return a str->float dict with extra things to log
        # This function has access to the collected batch and is optional
        return {}
