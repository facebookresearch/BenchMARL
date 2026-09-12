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

    # Dynamic Task Workload
    task_arrival_rate: float = MISSING
    task_data_min: float = MISSING
    task_data_max: float = MISSING
    cpu_cycles_per_bit: float = MISSING
    task_deadline: float = MISSING

    # MEC Server Capacity
    uav_num_cores: int = MISSING
    uav_cpu_freq: float = MISSING

    # Multi-Objective Reward Weights
    w_tasks: float = MISSING
    w_delay: float = MISSING
    w_energy: float = MISSING
    w_collision: float = MISSING
    w_sojourn: float = MISSING
