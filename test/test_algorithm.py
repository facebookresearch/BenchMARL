#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pytest

from benchmarl.algorithms import algorithm_config_registry
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.hydra_config import load_algorithm_config_from_hydra
from hydra import compose, initialize


@pytest.mark.parametrize("algo_name", algorithm_config_registry.keys())
def test_loading_algorithms(algo_name):
    with initialize(version_base=None, config_path="../benchmarl/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"algorithm={algo_name}",
                "task=vmas/balance",
            ],
        )
        algo_config: AlgorithmConfig = load_algorithm_config_from_hydra(cfg.algorithm)
        assert algo_config == algorithm_config_registry[algo_name].get_from_yaml()
