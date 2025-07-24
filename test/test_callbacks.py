#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest

from benchmarl.callbacks import callback_config_registry, LRSchedulerCallback
from benchmarl.hydra_config import load_callbacks_from_hydra

from hydra import compose, initialize


@pytest.mark.parametrize("callback_name", callback_config_registry.keys())
def test_loading_callbacks(callback_name):
    with initialize(version_base=None, config_path="../benchmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=mappo",
                "task=vmas/balance",
                f"callback@callbacks.c1={callback_name}",
            ],
        )
        callback = load_callbacks_from_hydra(cfg.callbacks)[0]
        assert isinstance(
            callback, callback_config_registry[callback_name].associated_class()
        )


class TestLRSchedulerCallback:
    callback_params_override = {
        "StepLR": [
            "scheduler_params={step_size: 2, gamma: 0.9}",
        ],
        "CosineAnnealingLR": [
            "scheduler_params={T_max: 1000}",
        ],
        "ExponentialLR": [
            "scheduler_params={gamma: 0.95}",
        ],
    }

    @pytest.mark.parametrize("scheduler_type", callback_params_override.keys())
    @pytest.mark.parametrize("target_optimizer", ["loss_objective", "loss_critic"])
    @pytest.mark.parametrize("log_lr", [True, False])
    def test_lr_scheduler(self, scheduler_type, target_optimizer, log_lr):
        """Test LR scheduler configuration creation with different parameters."""
        with initialize(version_base=None, config_path="../benchmarl/conf"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "algorithm=mappo",
                    "task=vmas/balance",
                    "callback@callbacks.c1=lr_scheduler",
                    f"callbacks.c1.scheduler_class=torch.optim.lr_scheduler.{scheduler_type}",
                    *[
                        f"++callbacks.c1.{override}"
                        for override in self.callback_params_override[scheduler_type]
                    ],
                    f"callbacks.c1.log_lr={log_lr}",
                ],
            )
            callback = load_callbacks_from_hydra(cfg.callbacks)[0]
            assert isinstance(callback, LRSchedulerCallback)
