#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from pathlib import Path

from benchmarl.hydra_config import reload_experiment_from_file

if __name__ == "__main__":

    # Let's assume that we have run an experiment with
    # `python benchmarl/run.py task=vmas/balance algorithm=mappo experiment.max_n_iters=2 experiment.on_policy_collected_frames_per_batch=100 experiment.checkpoint_interval=100`
    # and we have obtained
    # "outputs/2024-09-09/20-39-31/mappo_balance_mlp__cd977b69_24_09_09-20_39_31/checkpoints/checkpoint_100.pt""

    # Now we tell it where to restore from
    current_folder = Path(__file__).parent.absolute()
    restore_file = (
        current_folder
        / "outputs/2024-09-09/20-39-31/mappo_balance_mlp__cd977b69_24_09_09-20_39_31/checkpoints/checkpoint_100.pt"
    )

    experiment = reload_experiment_from_file(str(restore_file))
    experiment.evaluate()
