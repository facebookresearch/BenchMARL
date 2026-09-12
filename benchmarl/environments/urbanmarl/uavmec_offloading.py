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

    # MEC task workload and computing configuration
    task_arrival_rate: float = MISSING
    task_data_size_min: float = MISSING
    task_data_size_max: float = MISSING
    cpu_cycles_per_bit: float = MISSING
    max_latency_deadline: float = MISSING
    ue_tx_power: float = MISSING
    uav_cpu_freq: float = MISSING
    uav_num_cores: int = MISSING
    distance_weight: float = MISSING

    # Multi-objective reward weights
    w_tasks: float = MISSING
    w_latency: float = MISSING
    w_energy: float = MISSING
    w_collision: float = MISSING
    w_sojourn: float = MISSING
    extended_mec_obs: bool = MISSING
