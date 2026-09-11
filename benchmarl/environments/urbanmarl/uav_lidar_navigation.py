from dataclasses import dataclass, MISSING
from typing import List


@dataclass
class TaskConfig:
    num_uavs: int = MISSING
    num_ues: int = MISSING
    max_time_slots: int = MISSING
    volume_size: List[float] = MISSING
    max_horizontal_speed: float = MISSING
    max_vertical_speed: float = MISSING
    max_transmit_power: float = MISSING
    frequency_ghz: float = MISSING
    g2a_bandwidth: float = MISSING
    noise_figure_db: float = MISSING

    # LiDAR Sensor Configuration
    num_lidar_beams: int = MISSING
    lidar_max_range: float = MISSING

    # Reward Weights
    w_dist: float = MISSING
    w_reached: float = MISSING
    w_collision: float = MISSING
    w_proximity: float = MISSING
