"""Task configuration for the UrbanMARL uav_navigation scenario."""

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    num_uavs: int = MISSING
    num_ues: int = MISSING
    max_time_slots: int = MISSING
    volume_size: list[float] = MISSING
    max_horizontal_speed: float = MISSING
    max_vertical_speed: float = MISSING
    max_transmit_power: float = MISSING
    frequency_ghz: float = MISSING
    g2a_bandwidth: float = MISSING
    noise_figure_db: float = MISSING
