#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import hydra

from benchmarl.experiment import Callback, Experiment

from benchmarl.hydra_config import load_experiment_from_hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDictBase


class FootBallCurriculum(Callback):
    def __init__(self, n_frames_add_adversary):
        super().__init__()
        self.n_frames_add_adversary = n_frames_add_adversary
        self.activated = False

    def on_batch_collected(self, batch: TensorDictBase):
        if (
            self.experiment.total_frames > self.n_frames_add_adversary
            and not self.activated
        ):
            for scenario in [
                self.experiment.collector.env.scenario,
                self.experiment.test_env.scenario,
            ]:
                scenario.pos_shaping_factor_agent_ball = 0
                scenario.red_controller.enable()
            self.activated = True


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    experiment: Experiment = load_experiment_from_hydra(
        cfg,
        task_name=task_name,
        callbacks=[FootBallCurriculum(cfg.n_frames_add_adversary)],
    )

    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
