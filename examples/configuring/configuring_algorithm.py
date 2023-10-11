#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from benchmarl.algorithms import MasacConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    # Loads from "benchmarl/conf/algorithm/masac.yaml"
    algorithm_config = MasacConfig.get_from_yaml()

    # You can override from the script
    algorithm_config.num_qvalue_nets = 3  # Use an ensemble of 3 Q value nets
    algorithm_config.target_entropy = "auto"  # Set target entropy to auto
    algorithm_config.share_param_critic = True  # Use parameter sharing in the critic

    # Some basic other configs
    experiment_config = ExperimentConfig.get_from_yaml()
    task = VmasTask.BALANCE.get_from_yaml()
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
