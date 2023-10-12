#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#

python benchmarl/run.py task=vmas/balance algorithm=mappo model=layers/mlp model.layer_class="torch.nn.Linear" "model.num_cells=[32,32]" model.activation_class="torch.nn.ReLU"
