#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()
    # Loads from "benchmarl/conf/algorithm/mappo.yaml"
    algorithm_config = MappoConfig.get_from_yaml()
    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # Loading from yaml
    task1 = (
        VmasTask.BALANCE.get_from_yaml()
    )  # Get from yaml automatically converts to TaskClass
    task2 = (
        VmasTask.BALANCE.get_from_yaml()
    )  # Get from yaml automatically converts to TaskClass
    task1.config.update({"a": 1})
    task2.config.update({"a": 2})
    assert task1.config["a"] == 1
    assert task2.config["a"] == 2

    # You can pass either task1 or task1_immutable to the experiment

    experiment = Experiment(
        task=task1,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment.run()
