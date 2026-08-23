
from dataclasses import dataclass, MISSING

"""
    name: "urban_offloading"
    task_name: "DYNAMIC_OFFLOADING"
    num_uavs: 3
    num_ues: 50
    max_time_slots: 100
    volume_size: [500, 500, 200]
    max_horizontal_speed: 49.0
    max_vertical_speed: 12.0
    max_transmit_power: 5.0
    frequency_ghz: 29.0
    g2a_bandwidth: 10000000.0
    noise_figure_db: 7.0 
"""

@dataclass
class TaskConfig:
    # name : str = MISSING
    # task_name : str = MISSING
    # num_envs : int = MISSING
    # device : str = MISSING
    num_uavs : int = MISSING
    num_ues : int = MISSING
    max_time_slots : int = MISSING
    volume_size : list[float] = MISSING
    max_horizontal_speed : float = MISSING
    max_vertical_speed : float = MISSING
    max_transmit_power : float = MISSING
    frequency_ghz : float = MISSING
    g2a_bandwidth : float = MISSING
    noise_figure_db : float = MISSING