#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .cnn import Cnn, CnnConfig
from .common import (
    EnsembleModelConfig,
    Model,
    ModelConfig,
    SequenceModel,
    SequenceModelConfig,
)
from .deepsets import Deepsets, DeepsetsConfig
from .gnn import Gnn, GnnConfig
from .gru import Gru, GruConfig
from .lstm import Lstm, LstmConfig
from .mlp import Mlp, MlpConfig

classes = [
    "Mlp",
    "MlpConfig",
    "Gnn",
    "GnnConfig",
    "Cnn",
    "CnnConfig",
    "Deepsets",
    "DeepsetsConfig",
    "Gru",
    "GruConfig",
    "Lstm",
    "LstmConfig",
]

model_config_registry = {
    "mlp": MlpConfig,
    "gnn": GnnConfig,
    "cnn": CnnConfig,
    "deepsets": DeepsetsConfig,
    "gru": GruConfig,
    "lstm": LstmConfig,
}
