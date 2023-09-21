import os
import shutil
import uuid
from pathlib import Path

import pytest

from benchmarl.experiment import ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.common import ModelConfig, SequenceModelConfig
from torch import nn


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    folder_name = Path(os.path.dirname(os.path.realpath(__file__)))
    folder_name = folder_name / f"tmp_{str(uuid.uuid4())[:8]}"
    folder_name.mkdir(parents=False, exist_ok=False)
    os.chdir(folder_name)
    session._tmp_folder = folder_name


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    #"""
    shutil.rmtree(session._tmp_folder)


@pytest.fixture
def experiment_config() -> ExperimentConfig:
    experiment_config: ExperimentConfig = ExperimentConfig.get_from_yaml()
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
