#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pathlib

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase

from benchmarl.utils import class_from_name, DEVICE_TYPING, read_yaml_config


def _check_spec(tensordict, spec):
    if not spec.is_in(tensordict):
        raise ValueError(f"TensorDict {tensordict} not in spec {spec}")


def parse_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    del cfg["name"]
    kwargs = {}
    for key, value in cfg.items():
        if key.endswith("class") and value is not None:
            value = class_from_name(cfg[key])
        kwargs.update({key: value})
    return kwargs


def output_has_agent_dim(share_params: bool, centralised: bool) -> bool:
    """
    This is a dynamically computed attribute that indicates if the output will have the agent dimension.
    This will be false when share_params==True and centralised==True, and true in all other cases.
    When output_has_agent_dim is true, your model's output should contain the multiagent dimension,
    and the dimension should be absent otherwise

    """
    if share_params and centralised:
        return False
    else:
        return True


class Model(TensorDictModuleBase, ABC):
    """
    Abstract class representing a model.

    Models in BenchMARL are instantiated per agent group.
    This means that each model will process the inputs for a whole group of agents
    They are associated with input and output specs that define their domains.

    Args:
        input_spec (CompositeSpec): the input spec of the model
        output_spec (CompositeSpec): the output spec of the model
        agent_group (str): the name of the agent group the model is for
        n_agents (int): the number of agents this module is for
        device (str): the mdoel's device
        input_has_agent_dim (bool): This tells the model if the input will have a multi-agent dimension or not.
            For example, the input of policies will always have this set to true,
            but critics that use a global state have this set to false as the state is shared by all agents
        centralised (bool): This tells the model if it has full observability.
            This will always be true when self.input_has_agent_dim==False,
            but in cases where the input has the agent dimension, this parameter is
            used to distinguish between a decentralised model (where each agent's data
            is processed separately) and a centralized model, where the model pools all data together
        share_params (bool): This tells the model if it should have only one set of parameters
            or a different set of parameters for each agent.
            This is independent of the other options as it is possible to have different parameters
            for centralized critics with global input.
    """

    def __init__(
        self,
        input_spec: CompositeSpec,
        output_spec: CompositeSpec,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        **kwargs,
    ):
        TensorDictModuleBase.__init__(self)

        self.input_spec = input_spec
        self.output_spec = output_spec
        self.agent_group = agent_group
        self.input_has_agent_dim = input_has_agent_dim
        self.centralised = centralised
        self.share_params = share_params
        self.device = device
        self.n_agents = n_agents

        self.in_keys = list(self.input_spec.keys(True, True))
        self.out_keys = list(self.output_spec.keys(True, True))

        self.in_key = self.in_keys[0]
        self.out_key = self.out_keys[0]
        self.input_leaf_spec = self.input_spec[self.in_key]
        self.output_leaf_spec = self.output_spec[self.out_key]

        self._perform_checks()

    @property
    def output_has_agent_dim(self) -> bool:
        """
        This is a dynamically computed attribute that indicates if the output will have the agent dimension.
        This will be false when share_params==True and centralised==True, and true in all other cases.
        When output_has_agent_dim is true, your model's output should contain the multiagent dimension,
        and the dimension should be absent otherwise
        """
        return output_has_agent_dim(self.share_params, self.centralised)

    def _perform_checks(self):
        if not self.input_has_agent_dim and not self.centralised:
            raise ValueError(
                "If input does not have an agent dimension the model should be marked as centralised"
            )

        if len(self.in_keys) > 1:
            raise ValueError("Currently models support just one input key")
        if len(self.out_keys) > 1:
            raise ValueError("Currently models support just one output key")

        if self.agent_group in self.input_spec.keys() and self.input_spec[
            self.agent_group
        ].shape != (self.n_agents,):
            raise ValueError(
                "If the agent group is in the input specs, its shape should be the number of agents"
            )
        if self.agent_group in self.output_spec.keys() and self.output_spec[
            self.agent_group
        ].shape != (self.n_agents,):
            raise ValueError(
                "If the agent group is in the output specs, its shape should be the number of agents"
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # _check_spec(tensordict, self.input_spec)
        tensordict = self._forward(tensordict)
        # _check_spec(tensordict, self.output_spec)
        return tensordict

    ###############################
    # Abstract methods to implement
    ###############################

    @abstractmethod
    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Method to implement for the forward pass of the model.
        It should read self.in_key, process it and write self.out_key.

        Args:
            tensordict (TensorDictBase): the input td

        Returns: the input td with the written self.out_key

        """
        raise NotImplementedError


class SequenceModel(Model):
    def __init__(
        self,
        models: List[Model],
    ):

        super().__init__(
            n_agents=models[0].n_agents,
            input_spec=models[0].input_spec,
            output_spec=models[-1].output_spec,
            centralised=models[0].centralised,
            share_params=models[0].share_params,
            device=models[0].device,
            agent_group=models[0].agent_group,
            input_has_agent_dim=models[0].input_has_agent_dim,
        )
        self.models = TensorDictSequential(*models)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.models(tensordict)


@dataclass
class ModelConfig(ABC):
    """
    Dataclass representing an model configuration.
    This should be overridden by implemented models.
    Implementors should:
     1. add configuration parameters for their algorithm
     2. implement all abstract methods
    """

    def get_model(
        self,
        input_spec: CompositeSpec,
        output_spec: CompositeSpec,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
    ) -> Model:
        """
        Creates the model from the config.

        Args:
            input_spec (CompositeSpec): the input spec of the model
            output_spec (CompositeSpec): the output spec of the model
            agent_group (str): the name of the agent group the model is for
            n_agents (int): the number of agents this module is for
            device (str): the mdoel's device
            input_has_agent_dim (bool): This tells the model if the input will have a multi-agent dimension or not.
                For example, the input of policies will always have this set to true,
                but critics that use a global state have this set to false as the state is shared by all agents
            centralised (bool): This tells the model if it has full observability.
                This will always be true when self.input_has_agent_dim==False,
                but in cases where the input has the agent dimension, this parameter is
                used to distinguish between a decentralised model (where each agent's data
                is processed separately) and a centralized model, where the model pools all data together
            share_params (bool): This tells the model if it should have only one set of parameters
                or a different set of parameters for each agent.
                This is independent of the other options as it is possible to have different parameters
                for centralized critics with global input.

        Returns: the Model

        """
        return self.associated_class()(
            **asdict(self),
            input_spec=input_spec,
            output_spec=output_spec,
            agent_group=agent_group,
            input_has_agent_dim=input_has_agent_dim,
            n_agents=n_agents,
            centralised=centralised,
            share_params=share_params,
            device=device,
        )

    @staticmethod
    @abstractmethod
    def associated_class():
        """
        The associated Model class
        """
        raise NotImplementedError

    def process_env_fun(self, env_fun: Callable[[], EnvBase]) -> Callable[[], EnvBase]:
        """
        This function can be used to wrap env_fun
        Args:
            env_fun (callable): a function that takes no args and creates an enviornment

        Returns: a function that takes no args and creates an enviornment

        """
        return env_fun

    @staticmethod
    def _load_from_yaml(name: str) -> Dict[str, Any]:
        yaml_path = (
            pathlib.Path(__file__).parent.parent
            / "conf"
            / "model"
            / "layers"
            / f"{name.lower()}.yaml"
        )
        cfg = read_yaml_config(str(yaml_path.resolve()))
        return parse_model_config(cfg)

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        """
        Load the model configuration from yaml

        Args:
            path (str, optional): The full path of the yaml file to load from.
                If None, it will default to
                benchmarl/conf/model/layers/self.associated_class().__name__

        Returns: the loaded AlgorithmConfig
        """
        if path is None:
            return cls(
                **ModelConfig._load_from_yaml(
                    name=cls.associated_class().__name__,
                )
            )
        else:
            return cls(**parse_model_config(read_yaml_config(path)))


@dataclass
class SequenceModelConfig(ModelConfig):

    model_configs: Sequence[ModelConfig]
    intermediate_sizes: Sequence[int]

    def get_model(
        self,
        input_spec: CompositeSpec,
        output_spec: CompositeSpec,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
    ) -> Model:

        n_models = len(self.model_configs)
        if not n_models > 0:
            raise ValueError(
                f"SequenceModelConfig expects n_models > 0, got {n_models}"
            )
        if len(self.intermediate_sizes) != n_models - 1:
            raise ValueError(
                f"SequenceModelConfig intermediate_sizes len should be {n_models - 1}, got {len(self.intermediate_sizes)}"
            )

        out_has_agent_dim = output_has_agent_dim(share_params, centralised)
        next_centralised = not out_has_agent_dim
        intermediate_specs = [
            CompositeSpec(
                {
                    f"_{agent_group}_intermediate_{i}": UnboundedContinuousTensorSpec(
                        shape=(n_agents, size) if out_has_agent_dim else (size,)
                    )
                }
            )
            for i, size in enumerate(self.intermediate_sizes)
        ] + [output_spec]

        models = [
            self.model_configs[0].get_model(
                input_spec=input_spec,
                output_spec=intermediate_specs[0],
                agent_group=agent_group,
                input_has_agent_dim=input_has_agent_dim,
                n_agents=n_agents,
                centralised=centralised,
                share_params=share_params,
                device=device,
            )
        ]

        next_models = [
            self.model_configs[i].get_model(
                input_spec=intermediate_specs[i - 1],
                output_spec=intermediate_specs[i],
                agent_group=agent_group,
                input_has_agent_dim=out_has_agent_dim,
                n_agents=n_agents,
                centralised=next_centralised,
                share_params=share_params,
                device=device,
            )
            for i in range(1, n_models)
        ]
        models += next_models
        return SequenceModel(models)

    @staticmethod
    def associated_class():
        return SequenceModel

    def process_env_fun(self, env_fun: Callable[[], EnvBase]) -> Callable[[], EnvBase]:
        for model_config in self.model_configs:
            env_fun = model_config.process_env_fun(env_fun)
        return env_fun

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        raise NotImplementedError
