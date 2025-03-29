#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import importlib

import pytest

from benchmarl.experiment import ExperimentConfig
from benchmarl.models import CnnConfig, GnnConfig, GruConfig, LstmConfig, MlpConfig
from benchmarl.models.common import ModelConfig, SequenceModelConfig
from torch import nn

_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None
if _has_torch_geometric:
    import torch_geometric.nn.conv


@pytest.fixture
def experiment_config(tmp_path) -> ExperimentConfig:
    save_dir = tmp_path
    save_dir.mkdir(exist_ok=True)
    experiment_config: ExperimentConfig = ExperimentConfig.get_from_yaml()
    experiment_config.save_folder = str(save_dir)
    experiment_config.max_n_iters = 3
    experiment_config.max_n_frames = None

    experiment_config.on_policy_n_minibatch_iters = 1
    experiment_config.on_policy_minibatch_size = 2
    experiment_config.on_policy_collected_frames_per_batch = (
        experiment_config.off_policy_collected_frames_per_batch
    ) = 100
    experiment_config.on_policy_n_envs_per_worker = (
        experiment_config.off_policy_n_envs_per_worker
    ) = 2
    experiment_config.parallel_collection = False
    experiment_config.off_policy_n_optimizer_steps = 2
    experiment_config.off_policy_train_batch_size = 3
    experiment_config.off_policy_memory_size = 200

    experiment_config.evaluation = True
    experiment_config.render = True
    experiment_config.evaluation_episodes = 2
    experiment_config.evaluation_interval = 500
    experiment_config.evaluation_static = False
    experiment_config.loggers = ["csv"]
    experiment_config.create_json = True
    experiment_config.checkpoint_interval = 100
    return experiment_config


@pytest.fixture
def mlp_sequence_config() -> ModelConfig:
    return SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[8], activation_class=nn.Tanh, layer_class=nn.Linear),
            MlpConfig(num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear),
        ],
        intermediate_sizes=[5],
    )


@pytest.fixture
def cnn_sequence_config() -> ModelConfig:
    return SequenceModelConfig(
        model_configs=[
            CnnConfig(
                cnn_num_cells=[4, 3],
                cnn_kernel_sizes=[3, 2],
                cnn_strides=1,
                cnn_paddings=0,
                cnn_activation_class=nn.Tanh,
                mlp_num_cells=[4],
                mlp_activation_class=nn.Tanh,
                mlp_layer_class=nn.Linear,
            ),
            MlpConfig(num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear),
        ],
        intermediate_sizes=[5],
    )


@pytest.fixture
def mlp_gnn_sequence_config() -> ModelConfig:
    return SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[8], activation_class=nn.Tanh, layer_class=nn.Linear),
            GnnConfig(
                topology="full",
                self_loops=False,
                gnn_class=torch_geometric.nn.conv.GATv2Conv,
            ),
            MlpConfig(num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear),
        ],
        intermediate_sizes=[5, 3],
    )


@pytest.fixture
def cnn_gnn_sequence_config() -> ModelConfig:

    return SequenceModelConfig(
        model_configs=[
            CnnConfig(
                cnn_num_cells=[4, 3],
                cnn_kernel_sizes=[3, 2],
                cnn_strides=1,
                cnn_paddings=0,
                cnn_activation_class=nn.Tanh,
                mlp_num_cells=[4],
                mlp_activation_class=nn.Tanh,
                mlp_layer_class=nn.Linear,
            ),
            GnnConfig(
                topology="full",
                self_loops=False,
                gnn_class=torch_geometric.nn.conv.GATv2Conv,
            ),
            MlpConfig(num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear),
        ],
        intermediate_sizes=[5, 3],
    )


@pytest.fixture
def gru_mlp_sequence_config() -> ModelConfig:
    return SequenceModelConfig(
        model_configs=[
            GruConfig(
                hidden_size=13,
                mlp_num_cells=[],
                mlp_activation_class=nn.Tanh,
                mlp_layer_class=nn.Linear,
                n_layers=1,
                bias=True,
                dropout=0,
                compile=False,
            ),
            MlpConfig(num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear),
        ],
        intermediate_sizes=[5],
    )


@pytest.fixture
def lstm_mlp_sequence_config() -> ModelConfig:
    return SequenceModelConfig(
        model_configs=[
            LstmConfig(
                hidden_size=13,
                mlp_num_cells=[],
                mlp_activation_class=nn.Tanh,
                mlp_layer_class=nn.Linear,
                n_layers=1,
                bias=True,
                dropout=0,
                compile=False,
            ),
            MlpConfig(num_cells=[4], activation_class=nn.Tanh, layer_class=nn.Linear),
        ],
        intermediate_sizes=[5],
    )


@pytest.fixture
def cnn_lstm_sequence_config() -> ModelConfig:
    return SequenceModelConfig(
        model_configs=[
            CnnConfig(
                cnn_num_cells=[4, 3],
                cnn_kernel_sizes=[3, 2],
                cnn_strides=1,
                cnn_paddings=0,
                cnn_activation_class=nn.Tanh,
                mlp_num_cells=[4],
                mlp_activation_class=nn.Tanh,
                mlp_layer_class=nn.Linear,
            ),
            LstmConfig(
                hidden_size=13,
                mlp_num_cells=[],
                mlp_activation_class=nn.Tanh,
                mlp_layer_class=nn.Linear,
                n_layers=1,
                bias=True,
                dropout=0,
                compile=False,
            ),
        ],
        intermediate_sizes=[5],
    )
