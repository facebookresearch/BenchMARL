from typing import Any, Dict, Union

import torch
import yaml

DEVICE_TYPING = Union[torch.device, str, int]


# step 1: Read the file. Since file is small, we are doing a whole read.
def read_yaml_file(config_file: str) -> Dict[str, Any]:
    with open(config_file) as config:
        yaml_string = config.read()
    return yaml.safe_load(yaml_string)
