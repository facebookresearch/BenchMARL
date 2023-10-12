#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#

python benchmarl/run.py task=vmas/balance algorithm=mappo experiment.lr=0.03 experiment.evaluation=true experiment.train_device="cpu"
