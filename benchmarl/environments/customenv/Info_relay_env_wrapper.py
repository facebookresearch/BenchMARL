from typing import Dict, List, Tuple, Union
import warnings

from torchrl.envs.libs.pettingzoo import PettingZooWrapper, _has_pettingzoo
from torchrl.envs.utils import _classproperty, check_marl_grouping, MarlGroupMapType

### OBS this is only for using the PettingZooWrapper in our Info relat env - translate it to torchrl structs
class InfoRelayEnv(PettingZooWrapper):

    def __init__(
        self,
        task: str,
        parallel: bool,
        return_state: bool = False,
        group_map: MarlGroupMapType | Dict[str, List[str]] | None = None,
        use_mask: bool = False,
        categorical_actions: bool = True,
        seed: int | None = None,
        done_on_any: bool | None = None,
        **kwargs,
    ):
        if not _has_pettingzoo:
            raise ImportError(
                f"pettingzoo python package was not found. Please install this dependency. "
                f"More info: {self.git_url}."
            )
        kwargs["task"] = task
        kwargs["parallel"] = parallel
        kwargs["return_state"] = return_state
        kwargs["group_map"] = group_map
        kwargs["use_mask"] = use_mask
        kwargs["categorical_actions"] = categorical_actions
        kwargs["seed"] = seed
        kwargs["done_on_any"] = done_on_any

        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        if "task" not in kwargs:
            raise TypeError("Could not find environment key 'task' in kwargs.")
        if "parallel" not in kwargs:
            raise TypeError("Could not find environment key 'parallel' in kwargs.")

    def _build_env(
        self,
        task: str,
        parallel: bool,
        **kwargs,
    ) -> Union[
        "pettingzoo.utils.env.ParallelEnv",  # noqa: F821
        "pettingzoo.utils.env.AECEnv",  # noqa: F821
    ]:
        self.task_name = task

        try:
            from .Info_relay_env_v2 import Info_relay_env
        except ModuleNotFoundError as err:
            warnings.warn(
                f"Failed to load env with error message: {err}"
            )
            #print("Error")

        #if task != "info_relay":
        #    print("Only INFO_RELAY task can be run fom this command")


        ## OBS: here we can add vectorized envs: from torchrl.envs import ParallelEnv
        if parallel:
            petting_zoo_env = Info_relay_env(**kwargs)
        else:
            raise NotImplementedError

        return super()._build_env(env=petting_zoo_env)