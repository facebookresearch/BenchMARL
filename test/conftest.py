import pytest

from benchmarl.experiment import ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.common import ModelConfig, SequenceModelConfig
from torch import nn


@pytest.fixture
def experiment_config(tmp_path) -> ExperimentConfig:
    save_dir = tmp_path
    save_dir.mkdir(exist_ok=True)
    experiment_config: ExperimentConfig = ExperimentConfig.get_from_yaml()
    experiment_config.save_folder = str(save_dir)
    experiment_config.n_iters = 3
    experiment_config.n_optimizer_steps = 2
    experiment_config.n_envs_per_worker = 2
    experiment_config.collected_frames_per_batch = 100
    experiment_config.on_policy_minibatch_size = 10
    experiment_config.off_policy_memory_size = 200
    experiment_config.off_policy_train_batch_size = 100
    experiment_config.evaluation = True
    experiment_config.evaluation_episodes = 2
    experiment_config.loggers = ["csv"]
    experiment_config.create_json = True
    experiment_config.checkpoint_interval = 1
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
