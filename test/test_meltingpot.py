#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#


import packaging
import pytest
import torchrl

from benchmarl.algorithms import (
    algorithm_config_registry,
    IppoConfig,
    MasacConfig,
    QmixConfig,
    VdnConfig,
)
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import MeltingPotTask, Task
from benchmarl.experiment import Experiment

from utils import _has_meltingpot
from utils_experiment import ExperimentUtils


@pytest.mark.skipif(not _has_meltingpot, reason="Meltingpot not found")
@pytest.mark.skipif(
    packaging.version.parse(torchrl.__version__).base_version <= "0.3.1",
    reason="TorchRL <= 0.3.1 does nto support meltingpot",
)
class TestMeltingPot:
    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("task", [MeltingPotTask.COMMONS_HARVEST__OPEN])
    def test_all_algos(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        cnn_sequence_config,
    ):
        # To not run unsupported algo-task pairs
        if not algo_config.supports_discrete_actions():
            pytest.skip()

        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=cnn_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", [IppoConfig, MasacConfig])
    @pytest.mark.parametrize("task", list(MeltingPotTask))
    def test_all_tasks(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        cnn_sequence_config,
    ):
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=cnn_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()

    @pytest.mark.parametrize("algo_config", algorithm_config_registry.values())
    @pytest.mark.parametrize("task", [MeltingPotTask.COMMONS_HARVEST__OPEN])
    def test_reloading_trainer(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        experiment_config,
        cnn_sequence_config,
    ):
        # To not run unsupported algo-task pairs
        if not algo_config.supports_discrete_actions():
            pytest.skip()

        algo_config = algo_config.get_from_yaml()
        if isinstance(algo_config, VdnConfig):
            # There are some bugs currently in TorchRL https://github.com/pytorch/rl/issues/1593
            pytest.skip()
        ExperimentUtils.check_experiment_loading(
            algo_config=algo_config,
            model_config=cnn_sequence_config,
            experiment_config=experiment_config,
            task=task.get_from_yaml(),
        )

    @pytest.mark.parametrize("algo_config", [QmixConfig, IppoConfig, MasacConfig])
    @pytest.mark.parametrize("task", [MeltingPotTask.COMMONS_HARVEST__OPEN])
    @pytest.mark.parametrize("share_params", [True, False])
    def test_share_policy_params(
        self,
        algo_config: AlgorithmConfig,
        task: Task,
        share_params,
        experiment_config,
        cnn_sequence_config,
    ):
        experiment_config.share_policy_params = share_params
        task = task.get_from_yaml()
        experiment = Experiment(
            algorithm_config=algo_config.get_from_yaml(),
            model_config=cnn_sequence_config,
            seed=0,
            config=experiment_config,
            task=task,
        )
        experiment.run()
