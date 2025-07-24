#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pathlib

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from benchmarl.experiment import Callback
from benchmarl.utils import _read_yaml_config


@dataclass
class CallbackConfig:
    """
    Dataclass representing a callback configuration.
    This should be overridden by implemented callbacks.
    Implementors should:

        1. add configuration parameters for their callback
        2. implement all abstract methods

    """

    def get_callback(self) -> Callback:
        """
        Main function to turn the config into the associated callback

        Returns: the Callback

        """
        return self.associated_class()(
            **self.__dict__,  # Passes all the custom config parameters
        )

    @staticmethod
    def _load_from_yaml(name: str) -> Dict[str, Any]:
        yaml_path = (
            pathlib.Path(__file__).parent.parent
            / "conf"
            / "callbacks"
            / f"{name.lower()}.yaml"
        )
        return _read_yaml_config(str(yaml_path.resolve()))

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        """
        Load the callback configuration from yaml

        Args:
            path (str, optional): The full path of the yaml file to load from.
                If None, it will default to
                ``benchmarl/conf/callbacks/self.associated_class().__name__``

        Returns: the loaded CallbackConfig
        """

        if path is None:
            config = CallbackConfig._load_from_yaml(
                name=cls.associated_class().__name__
            )

        else:
            config = _read_yaml_config(path)
        return cls(**config)

    @staticmethod
    @abstractmethod
    def associated_class() -> Type[Callback]:
        """
        The callback class associated to the config
        """
        raise NotImplementedError
