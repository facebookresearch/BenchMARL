#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest

from benchmarl.callbacks import callback_config_registry
from benchmarl.hydra_config import load_callbacks_from_hydra

from hydra import compose, initialize


def test_no_callbacks():
    with initialize(version_base=None, config_path="../benchmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=mappo",
                "task=vmas/balance",
            ],
        )
        callbacks = load_callbacks_from_hydra(getattr(cfg, "callbacks", None) or {})
        assert len(callbacks) == 0


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


def test_disabling_callbacks():
    with initialize(version_base=None, config_path="../benchmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=mappo",
                "task=vmas/balance",
                "+callbacks=null",
            ],
        )
        callbacks = load_callbacks_from_hydra(getattr(cfg, "callbacks", None) or {})
        assert len(callbacks) == 0
