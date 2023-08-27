from benchmarl.algorithms.common import AlgorithmConfig, Algorithm

from benchmarl.models.common import ModelConfig, Model
from tensordict.nn import TensorDictModule
from torchrl import ReplayBuffer
from torchrl import LossModule

import time

import hydra
import torch

from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    AdditiveGaussianWrapper,
    ProbabilisticActor,
    TanhDelta,
    ValueOperator,
)
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from utils.logging import init_logging, log_evaluation, log_training
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.utils import NestedKey

from benchmarl.utils import DEVICE_TYPING


class Mappo(Algorithm):
    def __init__(self, share_param_actor: bool, share_param_critic: bool, **kwargs):
        super().__init__(**kwargs)

        self.share_param_actor = share_param_actor
        self.share_param_critic = share_param_critic

    def get_replay_buffer(
        self,
        memory_size: int,
        sampling_size: int,
        storing_device: DEVICE_TYPING,
    ) -> ReplayBuffer:
        return TensorDictReplayBuffer(
            storage=LazyTensorStorage(memory_size, device=storing_device),
            sampler=SamplerWithoutReplacement(),
            batch_size=sampling_size,
        )

    def get_loss(self, state_key: NestedKey = None) -> LossModule:
        pass
        # Policy
        # actor_net = nn.Sequential(
        #     self.model_config.get_model(n_agents=self.n_agents,input_features_shape=)
        #     NormalParamExtractor(),
        # )
        # policy_module = TensorDictModule(
        #     actor_net,
        #     in_keys=[("agents", "observation")],
        #     out_keys=[("agents", "loc"), ("agents", "scale")],
        # )
        # policy = ProbabilisticActor(
        #     module=policy_module,
        #     spec=env.unbatched_action_spec,
        #     in_keys=[("agents", "loc"), ("agents", "scale")],
        #     out_keys=[env.action_key],
        #     distribution_class=TanhNormal,
        #     distribution_kwargs={
        #         "min": env.unbatched_action_spec[("agents", "action")].space.minimum,
        #         "max": env.unbatched_action_spec[("agents", "action")].space.maximum,
        #     },
        #     return_log_prob=True,
        # )
        #
        # # Critic
        # module = MultiAgentMLP(
        #     n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        #     n_agent_outputs=1,
        #     n_agents=env.n_agents,
        #     centralised=cfg.model.centralised_critic,
        #     share_params=cfg.model.shared_parameters,
        #     device=cfg.train.device,
        #     depth=2,
        #     num_cells=256,
        #     activation_class=nn.Tanh,
        # )
        # value_module = ValueOperator(
        #     module=module,
        #     in_keys=[("agents", "observation")],
        # )

    def get_policy(self) -> TensorDictModule:
        pass

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False


class MappoConfig(AlgorithmConfig):
    # You can add any kwargs from benchmarl.algorithms.Mappo

    share_param_actor: bool = True
    share_param_critic: bool = True

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Mappo
