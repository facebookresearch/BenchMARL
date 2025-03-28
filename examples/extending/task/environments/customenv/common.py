#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDictBase

from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase
from torchrl.envs.libs import YourTorchRLEnvConstructor


class CustomEnvTask(Task):
    # Your task names.
    # Their config will be loaded from conf/task/customenv

    TASK_1 = None  # Loaded automatically from conf/task/customenv/task_1
    TASK_2 = None  # Loaded automatically from conf/task/customenv/task_2

    @staticmethod
    def associated_class():
        return CustomEnvClass


class CustomEnvClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        return lambda: YourTorchRLEnvConstructor(
            scenario=self.name.lower(),
            num_envs=num_envs,  # Number of vectorized envs (do not use this param if the env is not vectorized)
            continuous_actions=continuous_actions,  # Ignore this param if your env does not have this choice
            seed=seed,
            device=device,
            categorical_actions=True,  # If your env has discrete actions, they need to be categorical (TorchRL can help with this)
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
        return 100

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # The group map mapping group names to agent names
        # The data in the tensordict will havebe presented this way
        return {"agents": [agent.name for agent in env.agents]}

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the observation.
        # Must be a CompositeSpec with one (group_name, observation_key) entry per group.
        return env.full_observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        # A spec for the action.
        # If provided, must be a CompositeSpec with one (group_name, "action") entry per group.
        return env.full_action_spec

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the state.
        # If provided, must be a CompositeSpec with one "state" entry
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the action mask.
        # If provided, must be a CompositeSpec with one (group_name, "action_mask") entry per group.
        return None

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the info.
        # If provided, must be a CompositeSpec with one (group_name, "info") entry per group (this entry can be composite).
        return None

    @staticmethod
    def env_name() -> str:
        # The name of the environment in the benchmarl/conf/task folder
        return "customenv"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        # Optionally return a str->float dict with extra things to log
        # This function has access to the collected batch and is optional
        return {}
