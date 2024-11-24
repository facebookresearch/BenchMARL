#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import importlib

_has_vmas = importlib.util.find_spec("vmas") is not None
_has_smacv2 = importlib.util.find_spec("smacv2") is not None
_has_pettingzoo = importlib.util.find_spec("pettingzoo") is not None
_has_meltingpot = importlib.util.find_spec("meltingpot") is not None
_has_magent2 = importlib.util.find_spec("magent2") is not None
