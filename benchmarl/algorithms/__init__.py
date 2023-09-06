from .iddpg import Iddpg, IddpgConfig
from .ippo import Ippo, IppoConfig
from .maddpg import Maddpg, MaddpgConfig
from .mappo import Mappo, MappoConfig
from .masac import Masac, MasacConfig

all_algorithm_configs = (
    IppoConfig,
    MappoConfig,
    MaddpgConfig,
    IddpgConfig,
    MasacConfig,
)
