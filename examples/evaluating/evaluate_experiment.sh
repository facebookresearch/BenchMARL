#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#

# Assume we have run the experiment with
# python benchmarl/run.py task=vmas/balance algorithm=mappo experiment.max_n_iters=2 experiment.on_policy_collected_frames_per_batch=100 experiment.checkpoint_interval=100
# Evaluate it at the given checkpoint
python benchmarl/evaluate.py ../outputs/2024-09-09/20-39-31/mappo_balance_mlp__cd977b69_24_09_09-20_39_31/checkpoints/checkpoint_100.pt
