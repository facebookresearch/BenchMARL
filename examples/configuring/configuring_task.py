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
    # WARNING: Configuring tasks is only suggested for debugging.
    # For benchmarking, you should use the default configuration/

    # Loads from "benchmarl/conf/task/vmas/balance.yaml"
    task = VmasTask.BALANCE.get_from_yaml()

    # You can override from the script
    task.config["n_agents"] = 4  # Change the number of agents to 4

    # Some basic other configs
    algorithm_config = MappoConfig.get_from_yaml()
    experiment_config = ExperimentConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment.run()
