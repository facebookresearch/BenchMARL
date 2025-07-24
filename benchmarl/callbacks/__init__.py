#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import CallbackConfig
from .lr_scheduler import LRSchedulerCallback, LRSchedulerConfig

__all__ = [
    "CallbackConfig",
    "LRSchedulerCallback",
    "LRSchedulerConfig",
]

callback_config_registry = {
    "lr_scheduler": LRSchedulerConfig,
}
