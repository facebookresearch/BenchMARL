#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#

python benchmarl/run.py task=vmas/balance algorithm=mappo model=sequence "model.intermediate_sizes=[256]" "model/layers@model.layers.l1=mlp" "model/layers@model.layers.l2=mlp" "+model/layers@model.layers.l3=mlp" "model.layers.l3.num_cells=[3]"
