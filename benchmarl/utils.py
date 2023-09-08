from typing import Any, Dict, Union

import torch
import yaml

DEVICE_TYPING = Union[torch.device, str, int]


def read_yaml_config(config_file: str) -> Dict[str, Any]:
    with open(config_file) as config:
        yaml_string = config.read()
    config_dict = yaml.safe_load(yaml_string)
    if "defaults" in config_dict.keys():
        del config_dict["defaults"]
    return config_dict
