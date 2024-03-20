#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import Model, ModelConfig, SequenceModel, SequenceModelConfig
from .gnn import Gnn, GnnConfig
from .mlp import Mlp, MlpConfig
from .robomaster_model import RmGnn, RmGnnConfig

classes = ["Mlp", "MlpConfig", "Gnn", "GnnConfig", "RmGnn"]

model_config_registry = {"mlp": MlpConfig, "gnn": GnnConfig, "rmgnn": RmGnnConfig}
