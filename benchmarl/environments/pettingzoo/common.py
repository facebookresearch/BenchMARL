from typing import Callable, Dict, List, Optional

from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase, PettingZooEnv

from benchmarl.environments.common import Task


class PettingZooTask(Task):
    MULTIWALKER = None
    SIMPLE_TAG = None

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
    ) -> Callable[[], EnvBase]:
        if self.supports_continuous_actions() and self.supports_discrete_actions():
            self.config.update({"continuous_actions": continuous_actions})

        return lambda: PettingZooEnv(
            categorical_actions=True,
            seed=seed,
            parallel=True,
            return_state=self.has_state(),
            render_mode="rgb_array",
            **self.config
        )

    def supports_continuous_actions(self) -> bool:
        if self in {PettingZooTask.MULTIWALKER, PettingZooTask.SIMPLE_TAG}:
            return True
        return False

    def supports_discrete_actions(self) -> bool:
        if self in {PettingZooTask.SIMPLE_TAG}:
            return True
        return False

    def has_state(self) -> bool:
        if self in {PettingZooTask.SIMPLE_TAG}:
            return True
        return False

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> bool:
        return self.config["max_cycles"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        if "state" in env.observation_spec:
            return CompositeSpec({"state": env.observation_spec["state"].clone()})
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
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

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]

        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.input_spec["full_action_spec"]

    @staticmethod
    def env_name() -> str:
        return "pettingzoo"
