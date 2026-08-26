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
from benchmarl.environments import Task, UrbanEnvTask
from benchmarl.experiment import Experiment
from benchmarl.models import MlpConfig
from torch import nn
from utils import _has_urbanmarl
from utils_experiment import ExperimentUtils



@pytest.mark.skipif(not _has_urbanmarl, reason="UrbanMARL not found")
class TestUrbanMARL:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("prefer_continuous", [True, False])
    @pytest.mark.parametrize("task", [UrbanEnvTask.COVERAGE])
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
        # To not run unsupported algo-task pairs
        if (
            not task.supports_continuous_actions()
            and not algo_config.supports_discrete_actions()
        ) or (
            not task.supports_discrete_actions()
            and not algo_config.supports_continuous_actions()
        ):
            pytest.skip()
        
        experiment_config.prefer_continuous_actions = prefer_continuous
        experiment_config.render = False
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", [IppoConfig, MasacConfig])
    @pytest.mark.parametrize("task", list(UrbanEnvTask))
    def test_all_tasks(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        mlp_sequence_config,
    ):
        task = task.get_from_yaml()
        # To not run unsupported algo-task pairs
        if (
            not task.supports_continuous_actions()
            and not algo_config.supports_discrete_actions()
        ) or (
            not task.supports_discrete_actions()
            and not algo_config.supports_continuous_actions()
        ):
            pytest.skip()

        experiment_config.render = False
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
        task: Task = UrbanEnvTask.UAV_UE_LOS,
    ):
        task = task.get_from_yaml()
        experiment_config.render = False
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
        "algo_config", [IppoConfig, IsacConfig, IddpgConfig]
    )
    @pytest.mark.parametrize("task", [UrbanEnvTask.UAV_NAVIGATION])
    def test_gnn(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        mlp_gnn_sequence_config,
    ):
        task = task.get_from_yaml()
        # To not run unsupported algo-task pairs
        if (
            not task.supports_continuous_actions()
            and not algo_config.supports_discrete_actions()
        ) or (
            not task.supports_discrete_actions()
            and not algo_config.supports_continuous_actions()
        ):
            pytest.skip()
        
        experiment_config.render = False
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
    @pytest.mark.parametrize("task", [UrbanEnvTask.UAV_NAVIGATION])
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
        # To not run unsupported algo-task pairs
        if (
            not task.supports_continuous_actions()
            and not algo_config.supports_discrete_actions()
        ) or (
            not task.supports_discrete_actions()
            and not algo_config.supports_continuous_actions()
        ):
            pytest.skip()
        
        experiment_config.render = False
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
    @pytest.mark.parametrize("task", [UrbanEnvTask.UAV_NAVIGATION])
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
        experiment_config.render = False
        task = task.get_from_yaml()
        # To not run unsupported algo-task pairs
        if (
            not task.supports_continuous_actions()
            and not algo_config.supports_discrete_actions()
        ) or (
            not task.supports_discrete_actions()
            and not algo_config.supports_continuous_actions()
        ):
            pytest.skip()
        
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
    @pytest.mark.parametrize("task", [UrbanEnvTask.UAV_UE_LOS])
    def test_reloading_trainer(
        self,
        algo_config,
        task: Task,
        experiment_config,
        mlp_sequence_config,
    ):
        algo_config = algo_config.get_from_yaml()
        task = task.get_from_yaml()
        if (
            not task.supports_continuous_actions()
            and not algo_config.supports_discrete_actions()
        ) or (
            not task.supports_discrete_actions()
            and not algo_config.supports_continuous_actions()
        ):
            pytest.skip()

        experiment_config.render = False
        ExperimentUtils.check_experiment_loading(
            algo_config=algo_config,
            model_config=mlp_sequence_config,
            experiment_config=experiment_config,
            task=task,
        )

    @pytest.mark.parametrize(
        "algo_config", [QmixConfig, IppoConfig, MaddpgConfig, MasacConfig]
    )
    @pytest.mark.parametrize("task", [UrbanEnvTask.UAV_NAVIGATION])
    @pytest.mark.parametrize("share_params", [True, False])
    def test_share_policy_params(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        share_params,
        experiment_config,
        mlp_sequence_config,
    ):
        algo_config = algo_config.get_from_yaml()
        task = task.get_from_yaml()
        if (
            not task.supports_continuous_actions()
            and not algo_config.supports_discrete_actions()
        ) or (
            not task.supports_discrete_actions()
            and not algo_config.supports_continuous_actions()
        ):
            pytest.skip()

        experiment_config.render = False
        experiment_config.share_policy_params = share_params
        critic_model_config = MlpConfig(
            num_cells=[6], activation_class=nn.Tanh, layer_class=nn.Linear
        )
        experiment = Experiment(
            algorithm_config=algo_config,
            model_config=mlp_sequence_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()
        
    @pytest.mark.parametrize("algo_config", [MasacConfig])
    @pytest.mark.parametrize("task", list(UrbanEnvTask))
    def test_render(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        mlp_sequence_config,
    ):
        task = task.get_from_yaml()
        # To not run unsupported algo-task pairs
        if (
            not task.supports_continuous_actions()
            and not algo_config.supports_discrete_actions()
        ) or (
            not task.supports_discrete_actions()
            and not algo_config.supports_continuous_actions()
        ):
            pytest.skip()

        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=mlp_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

