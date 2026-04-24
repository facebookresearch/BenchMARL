from typing import Callable, Dict, List, Optional
import copy

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDictBase

from torchrl.data import Composite
from torchrl.envs import EnvBase
from torchrl.envs.libs import GymWrapper
from .GameTheoryEnv import TwoPlayerGameTheoryEnv

TWO_PLAYER_GAME_THEORY = "Two Player Game Theory"

class TwoPlayerGameTheoryTask(Task):
    # Your task names.
    # Their config will be loaded from conf/task/super_simple_env

    BATTLE_OF_SEXES   = None
    CHICKEN           = None
    PRISONERS_DILEMMA = None
    STAGE_HUNT        = None

    @staticmethod
    def associated_class():
        return TwoPlayerGameTheoryClass


class TwoPlayerGameTheoryClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        return lambda: TwoPlayerGameTheoryEnv(
            config['game_name'],
            seed=seed,
            device=device,
        )

    def supports_continuous_actions(self) -> bool:
        # Does the environment support continuous actions?
        return False

    def supports_discrete_actions(self) -> bool:
        # Does the environment support discrete actions?
        return True

    def has_render(self, env: EnvBase) -> bool:
        # Does the env have a env.render(mode="rgb_array") or env.render() function?
        return False

    def max_steps(self, env: EnvBase) -> int:
        # Maximum number of steps for a rollout during evaluation
        return 1

    # def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
    #     # The group map mapping group names to agent names
    #     # The data in the tensordict will havebe presented this way
    #     return {"agents": [agent.name for agent in env.agents]}

    def observation_spec(self, env: EnvBase) -> Composite:
        # A spec for the observation.
        # Must be a Composite with one (group_name, observation_key) entry per group.
        return env.full_observation_spec

    def action_spec(self, env: EnvBase) -> Composite:
        # A spec for the action.
        # If provided, must be a Composite with one (group_name, "action") entry per group.
        return env.full_action_spec

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        # A spec for the state.
        # If provided, must be a Composite with one "state" entry
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        # A spec for the action mask.
        # If provided, must be a Composite with one (group_name, "action_mask") entry per group.
        return None

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        # A spec for the info.
        # If provided, must be a Composite with one (group_name, "info") entry per group (this entry can be Composite).
        return None

    @staticmethod
    def env_name() -> str:
        # The name of the environment in the benchmarl/conf/task folder
        return TWO_PLAYER_GAME_THEORY

    @staticmethod
    def log_info(batch: TensorDictBase) -> Dict[str, float]:
        # Optionally return a str->float dict with extra things to log
        # This function has access to the collected batch and is optional
        return {}