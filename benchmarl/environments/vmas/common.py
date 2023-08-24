from typing import Optional
from benchmarl.environments.common import Task, load_config
from torchrl.envs import EnvBase
from torchrl.envs.libs.vmas import VmasEnv


class VmasTask(Task):
    BALANCE = load_config("balance")

    def get_env(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
    ) -> EnvBase:
        return VmasEnv(
            scenario=self.name.lower(),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            **self.config,
        )

    def supports_continuous_actions(self) -> bool:
        return True


if __name__ == "__main__":
    print(VmasTask.BALANCE.get_env(num_envs=2, continuous_actions=True, seed=0))
