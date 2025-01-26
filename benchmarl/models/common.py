#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pathlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from tensordict.utils import NestedKey
from torchrl.data import Composite, TensorSpec, Unbounded

from benchmarl.utils import _class_from_name, _read_yaml_config, DEVICE_TYPING


def _check_spec(tensordict, spec):
    if not spec.is_in(tensordict):
        raise ValueError(f"TensorDict {tensordict} not in spec {spec}")


def parse_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    del cfg["name"]
    kwargs = {}
    for key, value in cfg.items():
        if key.endswith("class") and value is not None:
            value = _class_from_name(cfg[key])
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
        input_spec (Composite): the input spec of the model
        output_spec (Composite): the output spec of the model
        agent_group (str): the name of the agent group the model is for
        n_agents (int): the number of agents this module is for
        device (str): the model's device
        input_has_agent_dim (bool): This tells the model if the input will have a multi-agent dimension or not.
            For example, the input of policies will always have this set to true,
            but critics that use a global state have this set to false as the state is shared by all agents
        centralised (bool): This tells the model if it has full observability.
            This will always be true when ``self.input_has_agent_dim==False``,
            but in cases where the input has the agent dimension, this parameter is
            used to distinguish between a decentralised model (where each agent's data
            is processed separately) and a centralized model, where the model pools all data together
        share_params (bool): This tells the model if it should have only one set of parameters
            or a different set of parameters for each agent.
            This is independent of the other options as it is possible to have different parameters
            for centralized critics with global input.
        action_spec (Composite): The action spec of the environment
        model_index (int): the index of the model in a sequence
        is_critic (bool): Whether the model is a critic
    """

    def __init__(
        self,
        input_spec: Composite,
        output_spec: Composite,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        action_spec: Composite,
        model_index: int,
        is_critic: bool,
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
        self.action_spec = action_spec
        self.model_index = model_index
        self.is_critic = is_critic

        self.in_keys = list(self.input_spec.keys(True, True))
        self.out_keys = list(self.output_spec.keys(True, True))

        self.out_key = self.out_keys[0]
        self.output_leaf_spec = self.output_spec[self.out_key]

        self._perform_checks()

    @property
    def output_has_agent_dim(self) -> bool:
        """
        This is a dynamically computed attribute that indicates if the output will have the agent dimension.
        This will be false when ``share_params==True and centralised==True``, and true in all other cases.
        When output_has_agent_dim is true, your model's output should contain the multi-agent dimension,
        and the dimension should be absent otherwise
        """
        return output_has_agent_dim(self.share_params, self.centralised)

    @property
    def in_key(self) -> NestedKey:
        if len(self.in_keys) > 1:
            raise ValueError("Model has more than one input key")
        return self.in_keys[0]

    @property
    def input_leaf_spec(self) -> TensorSpec:
        return self.input_spec[self.in_key]

    def _perform_checks(self):
        if not self.input_has_agent_dim and not self.centralised:
            raise ValueError(
                "If input does not have an agent dimension the model should be marked as centralised"
            )

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

    def share_params_with(self, other_model):
        """Share paramters with another identical model model.

        This function modifies in-place the parameters of ``other_model`` to reference the parameters of ``self``

        Args:
            other_model (Model): the model that will share the parameters of ``self``.

        """
        if (
            self.share_params != other_model.share_params
            or self.centralised != other_model.centralised
            or self.input_has_agent_dim != other_model.input_has_agent_dim
            or self.input_spec != other_model.input_spec
            or self.output_spec != other_model.output_spec
        ):
            warnings.warn(
                "Sharing parameters with models that are not identical. "
                "This might result in unintended behavior or error."
            )
        for param, other_param in zip(self.parameters(), other_model.parameters()):
            other_param.data[:] = param.data

    ###############################
    # Abstract methods to implement
    ###############################

    @abstractmethod
    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Method to implement for the forward pass of the model.
        It should read self.in_keys, process it and write self.out_key.

        Args:
            tensordict (TensorDictBase): the input td

        Returns: the input td with the written self.out_key

        """
        raise NotImplementedError


class SequenceModel(Model):
    """A sequence of :class:`~benchmarl.models.Model`

    Args:
       models (list of Model): the models in the sequence
    """

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
            action_spec=models[0].action_spec,
            model_index=models[0].model_index,
            is_critic=models[0].is_critic,
        )
        self.models = TensorDictSequential(*models)
        self.in_keys = self.models.in_keys
        self.out_keys = self.models.out_keys

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.models(tensordict)


@dataclass
class ModelConfig(ABC):
    """
    Dataclass representing a :class:`~benchmarl.models.Model` configuration.
    This should be overridden by implemented models.
    Implementors should:

        1. add configuration parameters for their algorithm
        2. implement all abstract methods

    """

    def get_model(
        self,
        input_spec: Composite,
        output_spec: Composite,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        action_spec: Composite,
        model_index: int = 0,
    ) -> Model:
        """
        Creates the model from the config.

        Args:
            input_spec (Composite): the input spec of the model
            output_spec (Composite): the output spec of the model
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
            action_spec (Composite): The action spec of the environment
            model_index (int): the index of the model in a sequence. Defaults to 0.

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
            action_spec=action_spec,
            model_index=model_index,
            is_critic=self.is_critic,
        )

    @staticmethod
    @abstractmethod
    def associated_class():
        """
        The associated Model class
        """
        raise NotImplementedError

    @property
    def is_rnn(self) -> bool:
        """
        Whether the model is an RNN
        """
        return False

    @property
    def is_critic(self):
        """
        Whether the model is a critic
        """
        if not hasattr(self, "_is_critic"):
            self._is_critic = False
        return self._is_critic

    @is_critic.setter
    def is_critic(self, value):
        """
        Set whether the model is a critic
        """
        self._is_critic = value

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        """Get additional specs needed by the model as input.

        This method is useful for adding recurrent states.

        The returned value should be key: spec with the desired ending shape.

        The batch and agent dimensions will automatically be added to the spec.

        Args:
            model_index (int, optional): the index of the model. Defaults to 0.

        """
        return Composite()

    def _get_model_state_spec_inner(
        self, model_index: int = 0, group: str = None
    ) -> Composite:
        return self.get_model_state_spec(model_index)

    @staticmethod
    def _load_from_yaml(name: str) -> Dict[str, Any]:
        yaml_path = (
            pathlib.Path(__file__).parent.parent
            / "conf"
            / "model"
            / "layers"
            / f"{name.lower()}.yaml"
        )
        return _read_yaml_config(str(yaml_path.resolve()))

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
            config = ModelConfig._load_from_yaml(name=cls.associated_class().__name__)
        else:
            config = _read_yaml_config(path)
        config = parse_model_config(config)
        return cls(**config)


@dataclass
class SequenceModelConfig(ModelConfig):
    """Dataclass for a :class:`~benchmarl.models.SequenceModel`.


    Examples:

          .. code-block:: python

            import torch_geometric
            from torch import nn
            from benchmarl.algorithms import IppoConfig
            from benchmarl.environments import VmasTask
            from benchmarl.experiment import Experiment, ExperimentConfig
            from benchmarl.models import SequenceModelConfig, GnnConfig, MlpConfig

            experiment = Experiment(
                algorithm_config=IppoConfig.get_from_yaml(),
                model_config=SequenceModelConfig(
                    model_configs=[
                        MlpConfig(num_cells=[8], activation_class=nn.Tanh, layer_class=nn.Linear),
                        GnnConfig(
                            topology="full",
                            self_loops=False,
                            gnn_class=torch_geometric.nn.conv.GraphConv,
                        ),
                        MlpConfig(num_cells=[6], activation_class=nn.Tanh, layer_class=nn.Linear),
                    ],
                    intermediate_sizes=[5, 3],
                ),
                seed=0,
                config=ExperimentConfig.get_from_yaml(),
                task=VmasTask.NAVIGATION.get_from_yaml(),
            )
            experiment.run()

    """

    model_configs: Sequence[ModelConfig]
    intermediate_sizes: Sequence[int]

    def __post_init__(self):
        for model_config in self.model_configs:
            if isinstance(model_config, EnsembleModelConfig):
                raise TypeError(
                    "SequenceModelConfig cannot contain EnsembleModelConfig layers, but the opposite can be done."
                )

    def get_model(
        self,
        input_spec: Composite,
        output_spec: Composite,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        action_spec: Composite,
        model_index: int = 0,
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
            Composite(
                {
                    f"_{agent_group}{'_critic' if self.is_critic else ''}_intermediate_{i}": Unbounded(
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
                action_spec=action_spec,
                model_index=0,
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
                action_spec=action_spec,
                model_index=i,
            )
            for i in range(1, n_models)
        ]
        models += next_models
        return SequenceModel(models)

    @staticmethod
    def associated_class():
        return SequenceModel

    @property
    def is_critic(self):
        if not hasattr(self, "_is_critic"):
            self._is_critic = False
        return self._is_critic

    @is_critic.setter
    def is_critic(self, value):
        self._is_critic = value
        for model_config in self.model_configs:
            model_config.is_critic = value

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        spec = Composite()
        for i, model_config in enumerate(self.model_configs):
            spec.update(model_config.get_model_state_spec(model_index=i))
        return spec

    @property
    def is_rnn(self) -> bool:
        is_rnn = False
        for model_config in self.model_configs:
            is_rnn += model_config.is_rnn
        return is_rnn

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        raise NotImplementedError


@dataclass
class EnsembleModelConfig(ModelConfig):

    model_configs_map: Dict[str, ModelConfig]

    def get_model(self, agent_group: str, **kwargs) -> Model:
        if agent_group not in self.model_configs_map.keys():
            raise ValueError(
                f"Environment contains agent group '{agent_group}' not present in the EnsembleModelConfig configuration."
            )
        return self.model_configs_map[agent_group].get_model(
            **kwargs, agent_group=agent_group
        )

    @staticmethod
    def associated_class():
        class EnsembleModel(Model):
            pass

        return EnsembleModel

    @property
    def is_critic(self):
        if not hasattr(self, "_is_critic"):
            self._is_critic = False
        return self._is_critic

    @is_critic.setter
    def is_critic(self, value):
        self._is_critic = value
        for model_config in self.model_configs_map.values():
            model_config.is_critic = value

    def _get_model_state_spec_inner(
        self, model_index: int = 0, group: str = None
    ) -> Composite:
        return self.model_configs_map[group].get_model_state_spec(
            model_index=model_index
        )

    @property
    def is_rnn(self) -> bool:
        is_rnn = False
        for model_config in self.model_configs_map.values():
            is_rnn += model_config.is_rnn
        return is_rnn

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        raise NotImplementedError
