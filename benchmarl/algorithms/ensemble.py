#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Type

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule

from torchrl.objectives import LossModule

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig

from benchmarl.models.common import ModelConfig


class EnsembleAlgorithm(Algorithm):
    def __init__(self, algorithms_map, **kwargs):
        super().__init__(**kwargs)
        self.algorithms_map = algorithms_map

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        return self.algorithms_map[group]._get_loss(group, policy_for_loss, continuous)

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        return self.algorithms_map[group]._get_parameters(group, loss)

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        return self.algorithms_map[group]._get_policy_for_loss(
            group, model_config, continuous
        )

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        return self.algorithms_map[group]._get_policy_for_collection(
            policy_for_loss, group, continuous
        )

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        return self.algorithms_map[group].process_batch(group, batch)

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        return self.algorithms_map[group].process_loss_vals(group, loss_vals)


@dataclass
class EnsembleAlgorithmConfig(AlgorithmConfig):

    algorithm_configs_map: Dict[str, AlgorithmConfig]

    def __post_init__(self):
        algorithm_configs = list(self.algorithm_configs_map.values())
        self._on_policy = algorithm_configs[0].on_policy()

        for algorithm_config in algorithm_configs[1:]:
            if algorithm_config.on_policy() != self._on_policy:
                raise ValueError(
                    "Algorithms in EnsembleAlgorithmConfig must either be all on_policy or all off_policy"
                )

        if (
            not self.supports_discrete_actions()
            and not self.supports_continuous_actions()
        ):
            raise ValueError(
                "Ensemble algorithm does not support discrete actions nor continuous actions."
                " Make sure that at least one type of action is supported across all the algorithms used."
            )

    def get_algorithm(self, experiment) -> Algorithm:
        if set(self.algorithm_configs_map.keys()) != set(experiment.group_map.keys()):
            raise ValueError(
                f"EnsembleAlgorithm group names {self.algorithm_configs_map.keys()} do not match "
                f"environment group names {experiment.group_map.keys()}"
            )
        return self.associated_class()(
            algorithms_map={
                group: algorithm_config.get_algorithm(experiment)
                for group, algorithm_config in self.algorithm_configs_map.items()
            },
            experiment=experiment,
        )

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        raise NotImplementedError

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return EnsembleAlgorithm

    def on_policy(self) -> bool:
        return self._on_policy

    def supports_continuous_actions(self) -> bool:
        supports_continuous_actions = True
        for algorithm_config in self.algorithm_configs_map.values():
            supports_continuous_actions *= (
                algorithm_config.supports_continuous_actions()
            )
        return supports_continuous_actions

    def supports_discrete_actions(self) -> bool:
        supports_discrete_actions = True
        for algorithm_config in self.algorithm_configs_map.values():
            supports_discrete_actions *= algorithm_config.supports_discrete_actions()
        return supports_discrete_actions

    def has_independent_critic(self) -> bool:
        has_independent_critic = False
        for algorithm_config in self.algorithm_configs_map.values():
            has_independent_critic += algorithm_config.has_independent_critic()
        return has_independent_critic

    def has_centralized_critic(self) -> bool:
        has_centralized_critic = False
        for algorithm_config in self.algorithm_configs_map.values():
            has_centralized_critic += algorithm_config.has_centralized_critic()
        return has_centralized_critic

    def has_critic(self) -> bool:
        return self.has_centralized_critic() or self.has_independent_critic()
