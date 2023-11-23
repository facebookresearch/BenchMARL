#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import Model, ModelConfig, SequenceModel, SequenceModelConfig
from .mlp import Mlp, MlpConfig

common = ["Model", "ModelConfig", "SequenceModel", "SequenceModelConfig"]
classes = [
    "Mlp",
    "MlpConfig",
]
__all__ = common + classes

model_config_registry = {"mlp": MlpConfig}
