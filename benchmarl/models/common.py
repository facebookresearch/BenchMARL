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
    if share_params and centralised:
        return False
    else:
        return True


class Model(TensorDictModuleBase, ABC):
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
        raise NotImplementedError

    def process_env_fun(self, env_fun: Callable[[], EnvBase]) -> Callable[[], EnvBase]:
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

    @staticmethod
    @abstractmethod
    def get_from_yaml(path: Optional[str] = None):
        raise NotImplementedError


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

    @staticmethod
    def get_from_yaml(path: Optional[str] = None):
        raise NotImplementedError
