#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .cnn import Cnn, CnnConfig
from .common import Model, ModelConfig, SequenceModel, SequenceModelConfig
from .gnn import Gnn, GnnConfig
from .mlp import Mlp, MlpConfig

classes = ["Mlp", "MlpConfig", "Gnn", "GnnConfig", "Cnn", "CnnConfig"]

model_config_registry = {"mlp": MlpConfig, "gnn": GnnConfig, "cnn": CnnConfig}
