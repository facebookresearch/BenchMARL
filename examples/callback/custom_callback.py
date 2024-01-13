#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import List

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.models.mlp import MlpConfig
from tensordict import TensorDict, TensorDictBase


class MyCallbackA(Callback):
    def on_batch_collected(self, batch: TensorDictBase):
        print(f"Callback A is doing something with the sampling batch {batch}")

    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        print(f"Callback A is computing a loss with the training tensordict {batch}")
        return TensorDict({}, [])

    def on_train_end(self, training_td: TensorDictBase, group: str):
        print(
            f"Callback A is doing something with the training tensordict {training_td}"
        )

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        print(f"Callback A is doing something with the evaluation rollouts {rollouts}")


class MyCallbackB(Callback):
    def on_setup(self):
        print("Callback B is being called during setup")

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        print(
            "Callback B just reminds you that you fo not need to implement all methods and"
            f"you always have access to the experiment {self.experiment} and all its contents"
            f"like the policy {self.experiment.policy}"
        )


if __name__ == "__main__":
    experiment_config = ExperimentConfig.get_from_yaml()
    task = VmasTask.BALANCE.get_from_yaml()
    algorithm_config = MappoConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
        callbacks=[MyCallbackA(), MyCallbackB()],
    )
    experiment.run()
