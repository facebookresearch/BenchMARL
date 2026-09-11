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

    # Dynamic Ground User Mobility Parameters
    mobility_model: str = MISSING
    ue_speed_min: float = MISSING
    ue_speed_max: float = MISSING

    # Reward Weights
    w_los: float = MISSING
    w_collision: float = MISSING
    w_sojourn: float = MISSING
