#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import importlib
import random
from typing import Any, Dict, Union

import torch
import yaml

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
    torch.cuda.manual_seed_all(seed)
    if _has_numpy:
        import numpy

        numpy.random.seed(seed)
