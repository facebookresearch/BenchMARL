from .iddpg import Iddpg, IddpgConfig
from .ippo import Ippo, IppoConfig
from .iql import Iql, IqlConfig
from .isac import Isac, IsacConfig
from .maddpg import Maddpg, MaddpgConfig
from .mappo import Mappo, MappoConfig
from .masac import Masac, MasacConfig
from .qmix import Qmix, QmixConfig
from .vdn import Vdn, VdnConfig

all_algorithm_configs = (
    IppoConfig,
    MappoConfig,
    MaddpgConfig,
    IddpgConfig,
    MasacConfig,
    IsacConfig,
    QmixConfig,
    VdnConfig,
    IqlConfig,
)
