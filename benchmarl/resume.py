#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import argparse
from pathlib import Path

from experiment import Experiment

from benchmarl.hydra_config import reload_experiment_from_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resumes the experiment from a checkpoint file."
    )
    parser.add_argument(
        "checkpoint_file", type=str, help="The name of the checkpoint file"
    )
    args = parser.parse_args()
    checkpoint_file = str(Path(args.checkpoint_file).resolve())

    try:
        experiment = reload_experiment_from_file(checkpoint_file)
    except ValueError:
        experiment = Experiment.reload_from_file(checkpoint_file)
    experiment.run()
