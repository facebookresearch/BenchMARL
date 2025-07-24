#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING
from typing import Any, Dict

from tensordict import TensorDictBase

from benchmarl.callbacks.common import CallbackConfig
from benchmarl.experiment.callback import Callback
from benchmarl.utils import _class_from_name


@dataclass
class LRSchedulerConfig(CallbackConfig):
    """Configuration for the LR Scheduler Callback."""

    scheduler_class: str = MISSING  # "torch.optim.lr_scheduler.StepLR"
    scheduler_params: Dict[str, Any] = MISSING
    log_lr: bool = MISSING

    @staticmethod
    def associated_class():
        return LRSchedulerCallback


class LRSchedulerCallback(Callback):
    """
    Callback that applies learning rate scheduling to a single optimizer.

    Uses PyTorch's built-in schedulers.
    """

    def __init__(
        self,
        scheduler_class: str,
        scheduler_params: Dict[str, Any],
        log_lr: bool = True,
    ):
        super().__init__()
        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params
        self.log_lr = log_lr

        self.schedulers = None
        self.initial_logging = False

    def on_setup(self):
        """Setup the scheduler after the experiment is initialized."""

        scheduler_class = _class_from_name(self.scheduler_class)
        kwargs = {
            k: v
            for k, v in self.scheduler_params.items()
            if k in scheduler_class.__init__.__code__.co_varnames
        }

        self.schedulers = {}
        for group in self.experiment.optimizers:
            self.schedulers[group] = []
            for optimizer in self.experiment.optimizers[group].values():
                scheduler = scheduler_class(optimizer, **kwargs)
                self.schedulers[group].append(scheduler)

    def on_batch_collected(self, batch: TensorDictBase):
        if self.log_lr and not self.initial_logging:
            to_log = {
                f"train/{group}/lr": self.schedulers[group][0].get_last_lr()[0]
                for group in self.experiment.group_map.keys()
            }
            self.experiment.logger.log(to_log, step=self.experiment.n_iters_performed)
            self.initial_logging = True

    def on_train_end(self, training_td: TensorDictBase, group: str):
        """Step the scheduler after each collection step."""
        for scheduler in self.schedulers[group]:
            scheduler.step()

        if self.log_lr:
            lr = self.schedulers[group][0].get_last_lr()[0]
            to_log = {f"train/{group}/lr": lr}
            self.experiment.logger.log(to_log, step=self.experiment.n_iters_performed)

        return None
