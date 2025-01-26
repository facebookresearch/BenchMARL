#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import Algorithm, AlgorithmConfig
from .ensemble import EnsembleAlgorithm, EnsembleAlgorithmConfig
from .iddpg import Iddpg, IddpgConfig
from .ippo import Ippo, IppoConfig
from .iql import Iql, IqlConfig
from .isac import Isac, IsacConfig
from .maddpg import Maddpg, MaddpgConfig
from .mappo import Mappo, MappoConfig
from .masac import Masac, MasacConfig
from .qmix import Qmix, QmixConfig
from .vdn import Vdn, VdnConfig

classes = [
    "Iddpg",
    "IddpgConfig",
    "Ippo",
    "IppoConfig",
    "Iql",
    "IqlConfig",
    "Isac",
    "IsacConfig",
    "Maddpg",
    "MaddpgConfig",
    "Mappo",
    "MappoConfig",
    "Masac",
    "MasacConfig",
    "Qmix",
    "QmixConfig",
    "Vdn",
    "VdnConfig",
]

# A registry mapping "algoname" to its config dataclass
# This is used to aid loading of algorithms from yaml
algorithm_config_registry = {
    "mappo": MappoConfig,
    "ippo": IppoConfig,
    "maddpg": MaddpgConfig,
    "iddpg": IddpgConfig,
    "masac": MasacConfig,
    "isac": IsacConfig,
    "qmix": QmixConfig,
    "vdn": VdnConfig,
    "iql": IqlConfig,
}
