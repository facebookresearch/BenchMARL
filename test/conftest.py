import pytest

from benchmarl.experiment import ExperimentConfig


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
    experiment_config.evaluation = False
    experiment_config.loggers = []
    experiment_config.create_json = False
    experiment_config.checkpoint_interval = 0
    return experiment_config
