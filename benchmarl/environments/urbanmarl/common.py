from typing import Callable, Dict, List, Optional

from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.envs import EnvBase

#
from urbanmarl.envs.base_env import UrbanEnv
from urbanmarl.scenarios import load_scenario

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING


class UrbanEnvClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        return lambda: UrbanEnv(
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            scenario=self.name.lower(),
            # scenario=self.config.get("scenario", "default"),  # read from config
            **self.config,
        )

    def supports_continuous_actions(self) -> bool:
        scenario = load_scenario(self.name.lower(), self.config)
        return scenario.continuous_actions

    def supports_discrete_actions(self) -> bool:
        scenario = load_scenario(self.name.lower(), self.config)
        return scenario.discrete_actions

    def has_render(self, env: EnvBase) -> bool:
        return True

    @staticmethod
    def render_callback(experiment, env: UrbanEnv, data: TensorDictBase):
        """
        BenchMARL callback for rendering during evaluation.
        Called at every step during evaluation to provide
        pixels for video logging.
        """
        img = env.scenario.render(
            env, algorithm=experiment.algorithm_name, mode="rgb_array"
        )
        # img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)
        return img

    def max_steps(self, env: EnvBase) -> int:
        return env.max_steps

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.full_observation_spec_unbatched

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        if "state" in env.full_observation_spec_unbatched.keys():
            return Composite({"state": env.full_observation_spec_unbatched["state"]})
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    @staticmethod
    def env_name() -> str:
        return "urbanmarl"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        if "info" not in batch.keys():
            return {}
        if ("info", "urban_params") not in batch.keys(True, True):
            return {}
        info = {}
        for i in range(batch.batch_size[0]):
            alpha, beta, gamma, E = batch.get(("next", "info", "urban_params"))[i, 0]
            urban_name = f"{alpha.item():.2f}_{int(beta.item())}_{gamma.item():.2f}_{E.item():.4f}"
            for key in batch.keys(True, True):
                if isinstance(key, tuple) and key[0] == "info":
                    if key[0] == "info" and key[-1] == "urban_params":
                        continue
                    #
                    metric = key[-1]
                    metric_name = f"{metric}_{urban_name}"
                    info[metric_name] = batch.get(key)[i].mean()
        return info


class UrbanEnvTask(Task):
    UAV_NAVIGATION = None
    UAV_UE_LOS = None
    UAVMEC_OFFLOADING = None
    COVERAGE = None
    UAV_MOBILE_UE = None
    UAV_LIDAR_NAVIGATION = None
    UAVMEC_ADVANCED_PHYSICS = None
    MEC_OFFLOADING = None

    @staticmethod
    def associated_class():
        return UrbanEnvClass
