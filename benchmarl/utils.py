#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import contextlib
import importlib
import random
import typing
from typing import Any, Callable, Dict, List, Union

import torch
import yaml
from torchrl.data import Composite
from torchrl.envs import Compose, EnvBase, InitTracker, TensorDictPrimer, TransformedEnv

if typing.TYPE_CHECKING:
    from benchmarl.models import ModelConfig

_has_numpy = importlib.util.find_spec("numpy") is not None


DEVICE_TYPING = Union[torch.device, str, int]


def _read_yaml_config(config_file: str) -> Dict[str, Any]:
    with open(config_file) as config:
        yaml_string = config.read()
    config_dict = yaml.safe_load(yaml_string)
    if config_dict is None:
        config_dict = {}
    if "defaults" in config_dict.keys():
        del config_dict["defaults"]
    return config_dict


def _class_from_name(name: str):
    name_split = name.split(".")
    module_name = ".".join(name_split[:-1])
    class_name = name_split[-1]
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if _has_numpy:
        import numpy

        numpy.random.seed(seed)


@contextlib.contextmanager
def local_seed():
    torch_state = torch.random.get_rng_state()
    if _has_numpy:
        import numpy as np

        np_state = np.random.get_state()
    py_state = random.getstate()

    yield

    torch.random.set_rng_state(torch_state)
    if _has_numpy:
        np.random.set_state(np_state)
    random.setstate(py_state)


def _add_rnn_transforms(
    env_fun: Callable[[], EnvBase],
    group_map: Dict[str, List[str]],
    model_config: "ModelConfig",
) -> Callable[[], EnvBase]:
    """
    This function adds RNN specific transforms to the environment

    Args:
        env_fun (callable): a function that takes no args and creates an environment
        group_map (Dict[str,List[str]]): the group_map of the agents
        model_config (ModelConfig): the model configuration

    Returns: a function that takes no args and creates an environment

    """

    def model_fun():
        env = env_fun()
        spec_actor = Composite(
            {
                group: Composite(
                    model_config._get_model_state_spec_inner(group=group).expand(
                        len(agents),
                        *model_config._get_model_state_spec_inner(group=group).shape
                    ),
                    shape=(len(agents),),
                )
                for group, agents in group_map.items()
            }
        )

        out_env = TransformedEnv(
            env,
            Compose(
                *(
                    [InitTracker(init_key="is_init")]
                    + (
                        [
                            TensorDictPrimer(
                                spec_actor, reset_key="_reset", expand_specs=True
                            )
                        ]
                        if len(spec_actor.keys(True, True)) > 0
                        else []
                    )
                )
            ),
        )
        return out_env

    return model_fun
