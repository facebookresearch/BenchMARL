#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .gnn import GnnConfig
from .mlp import MlpConfig

model_config_registry = {"mlp": MlpConfig, "gnn": GnnConfig}
