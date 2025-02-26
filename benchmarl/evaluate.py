#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import argparse
from pathlib import Path

from experiment import Experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates the experiment from a checkpoint file."
    )
    parser.add_argument(
        "checkpoint_file", type=str, help="The name of the checkpoint file"
    )
    args = parser.parse_args()
    checkpoint_file = str(Path(args.checkpoint_file).resolve())

    experiment = Experiment.reload_from_file(checkpoint_file)

    experiment.evaluate()
