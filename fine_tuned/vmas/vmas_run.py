#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import hydra
from benchmarl.experiment import Callback, Experiment

from benchmarl.hydra_config import load_experiment_from_hydra
from benchmarl.models.robomaster_model import RmGnn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def get_model(model):
    model = model.module[0]
    while not isinstance(model, RmGnn):
        model = model[0]
    return model


def _share_params(model, other_model):
    if isinstance(model, list):
        for inner_model, inner_other_model in zip(model, other_model):
            for param, other_param in zip(
                inner_model.parameters(), inner_other_model.parameters()
            ):
                other_param.data[:] = param.data
    else:
        for param, other_param in zip(model.parameters(), other_model.parameters()):
            other_param.data[:] = param.data


class ShareActroCriticParams(Callback):
    def _share_params(self):
        policy = list(self.experiment.group_policies.values())[0]

        policy_model = get_model(policy)
        critic_model = list(self.experiment.algorithm.group_critics.values())[0]

        policy_model_gnn = policy_model.gnns
        critic_model_gnn = critic_model.gnns

        _share_params(policy_model_gnn, critic_model_gnn)

        policy_model_mlp = list(policy_model.mlp_local_and_comms.agent_networks[0])[:-2]
        critic_model_mlp = list(critic_model.mlp_local_and_comms.agent_networks[0])[:-2]

        _share_params(policy_model_mlp, critic_model_mlp)

    def on_setup(self):
        self._share_params()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    experiment: Experiment = load_experiment_from_hydra(
        cfg, task_name=task_name, callbacks=[ShareActroCriticParams()]
    )
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
