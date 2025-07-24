#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest
from benchmarl.algorithms import (
    algorithm_config_registry,
    IddpgConfig,
    IppoConfig,
    IsacConfig,
    MaddpgConfig,
    MappoConfig,
    MasacConfig,
    QmixConfig,
)
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task, VmasTask
from benchmarl.experiment import Experiment

from benchmarl.hydra_config import load_callbacks_from_hydra
from benchmarl.models import MlpConfig
from hydra import compose, initialize
from torch import nn
from utils import _has_vmas
from utils_experiment import ExperimentUtils


@pytest.mark.skipif(not _has_vmas, reason="VMAS not found")
class TestVmas:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("prefer_continuous", [True, False])
    @pytest.mark.parametrize("task", [VmasTask.BALANCE])
    def test_all_algos(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        prefer_continuous,
        experiment_config,
        mlp_sequence_config,
    ):
        # To not run the same test twice
        if (prefer_continuous and not algo_config.supports_continuous_actions()) or (
            not prefer_continuous and not algo_config.supports_discrete_actions()
        ):
            pytest.skip()

        task = task.get_from_yaml()
        experiment_config.prefer_continuous_actions = prefer_continuous
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", [IppoConfig, MasacConfig])
    @pytest.mark.parametrize("task", list(VmasTask))
    def test_all_tasks(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        mlp_sequence_config,
    ):
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    def test_collect_with_grad(
        self,
        experiment_config,
        mlp_sequence_config,
        algo_config: AlgorithmConfig = IppoConfig,
        task: Task = VmasTask.BALANCE,
    ):
        task = task.get_from_yaml()
        experiment_config.collect_with_grad = True
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize(
        "algo_config", [IppoConfig, QmixConfig, IsacConfig, IddpgConfig]
    )
    @pytest.mark.parametrize("task", [VmasTask.NAVIGATION])
    def test_gnn(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        mlp_gnn_sequence_config,
    ):
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_gnn_sequence_config,
            critic_model_config=mlp_gnn_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize(
        "algo_config", [MaddpgConfig, IppoConfig, QmixConfig, MasacConfig]
    )
    @pytest.mark.parametrize("task", [VmasTask.NAVIGATION])
    def test_gru(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        gru_mlp_sequence_config,
        share_params: bool = False,
    ):
        algo_config = algo_config.get_from_yaml()
        if algo_config.has_critic():
            algo_config.share_param_critic = share_params
        experiment_config.share_policy_params = share_params
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=gru_mlp_sequence_config,
            critic_model_config=gru_mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize(
        "algo_config", [IddpgConfig, MappoConfig, QmixConfig, IsacConfig]
    )
    @pytest.mark.parametrize("task", [VmasTask.NAVIGATION])
    def test_lstm(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        lstm_mlp_sequence_config,
        share_params: bool = False,
    ):
        algo_config = algo_config.get_from_yaml()
        if algo_config.has_critic():
            algo_config.share_param_critic = share_params
        experiment_config.share_policy_params = share_params
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=lstm_mlp_sequence_config,
            critic_model_config=lstm_mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("task", [VmasTask.BALANCE])
    def test_reloading_trainer(
        self,
        algo_config,
        task: Task,
        experiment_config,
        mlp_sequence_config,
    ):
        algo_config = algo_config.get_from_yaml()

        ExperimentUtils.check_experiment_loading(
            algo_config=algo_config,
            model_config=mlp_sequence_config,
            experiment_config=experiment_config,
            task=task.get_from_yaml(),
        )

    @pytest.mark.parametrize(
        "algo_config", [QmixConfig, IppoConfig, MaddpgConfig, MasacConfig]
    )
    @pytest.mark.parametrize("task", [VmasTask.NAVIGATION])
    @pytest.mark.parametrize("share_params", [True, False])
    def test_share_policy_params(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        share_params,
        experiment_config,
        mlp_sequence_config,
    ):
        experiment_config.share_policy_params = share_params
        critic_model_config = MlpConfig(
            num_cells=[6], activation_class=nn.Tanh, layer_class=nn.Linear
        )
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()


@pytest.mark.skipif(not _has_vmas, reason="VMAS not found")
class TestLRSchedulerCallbackInVmas:
    callback_params_override = {
        "StepLR": [
            "scheduler_params={step_size: 2, gamma: 0.9}",
        ],
        "CosineAnnealingLR": [
            "scheduler_params={T_max: 1000}",
        ],
        "ExponentialLR": [
            "scheduler_params={gamma: 0.95}",
        ],
    }

    def _get_callbacks(self, scheduler_type):
        with initialize(version_base=None, config_path="../benchmarl/conf"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "algorithm=mappo",
                    "task=vmas/balance",
                    "callback@callbacks.c1=lr_scheduler",
                    f"callbacks.c1.scheduler_class=torch.optim.lr_scheduler.{scheduler_type}",
                    *[
                        f"++callbacks.c1.{override}"
                        for override in self.callback_params_override[scheduler_type]
                    ],
                    "callbacks.c1.log_lr=True",
                ],
            )
            callbacks = load_callbacks_from_hydra(cfg.callbacks)
            return callbacks

    def _get_lr(self, experiment: Experiment):
        return next(
            iter(next(iter(experiment.callbacks[0].schedulers.values())).values())
        ).get_last_lr()[0]

    @pytest.mark.parametrize("scheduler_type", callback_params_override.keys())
    def test_lr_scheduler(
        self,
        scheduler_type,
        experiment_config,
        mlp_sequence_config,
    ):
        """Test LR scheduler configuration creation with different parameters."""
        callbacks = self._get_callbacks(scheduler_type)

        experiment = Experiment(
            algorithm_config=MappoConfig.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=VmasTask.NAVIGATION.get_from_yaml(),
            callbacks=callbacks,
        )
        experiment.run()

    @pytest.mark.parametrize("scheduler_type", callback_params_override.keys())
    def test_reloading_trainer(
        self,
        scheduler_type,
        experiment_config,
        mlp_sequence_config,
    ):
        algorithm_config = MappoConfig.get_from_yaml()
        task = VmasTask.NAVIGATION.get_from_yaml()

        max_n_iters = experiment_config.max_n_iters
        experiment = Experiment(
            algorithm_config=algorithm_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
            callbacks=self._get_callbacks(scheduler_type),
        )
        initial_lr = self._get_lr(experiment)
        experiment.run()
        exp_folder = experiment.folder_name
        end_lr = self._get_lr(experiment)
        assert initial_lr != end_lr

        experiment_config.max_n_iters = max_n_iters + 3
        experiment_config.restore_file = (
            exp_folder / "checkpoints" / f"checkpoint_{experiment.total_frames}.pt"
        )
        experiment_config.save_folder = None
        experiment = Experiment(
            algorithm_config=algorithm_config,
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
            callbacks=self._get_callbacks(scheduler_type),
        )
        reloaded_lr = self._get_lr(experiment)
        assert reloaded_lr == end_lr
