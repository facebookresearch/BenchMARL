from typing import Callable, Dict, List, Optional

from tensordict import TensorDictBase
from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase
from torchrl.envs.libs.smacv2 import SMACv2Env

from benchmarl.environments.common import Task


class Smacv2Task(Task):
    protoss_5_vs_5 = None

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
    ) -> Callable[[], EnvBase]:

        return lambda: SMACv2Env(categorical_actions=True, seed=seed, **self.config)

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> bool:
        return env.episode_limit

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        del observation_spec["info"]
        del observation_spec["agents"]
        return observation_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        del observation_spec["info"]
        del observation_spec["state"]
        del observation_spec[("agents", "observation")]
        return observation_spec

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        observation_spec = env.observation_spec.clone()
        del observation_spec["info"]
        del observation_spec["state"]
        del observation_spec[("agents", "action_mask")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        del observation_spec["state"]
        del observation_spec["agents"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.input_spec["full_action_spec"]

    @staticmethod
    def log_info(batch: TensorDictBase) -> Dict:
        done = batch.get(("next", "done")).squeeze(-1)
        return {
            "win_rate": batch.get(("next", "info", "battle_won"))[done].mean().item(),
            "episode_limit_rate": batch.get(("next", "info", "episode_limit"))[done]
            .mean()
            .item(),
        }

    @staticmethod
    def env_name() -> str:
        return "smacv2"


if __name__ == "__main__":
    print(Smacv2Task.protoss_5_vs_5.get_from_yaml())
    env = Smacv2Task.protoss_5_vs_5.get_env_fun(0, False, 0)()
    print(env.render(mode="rgb_array"))
