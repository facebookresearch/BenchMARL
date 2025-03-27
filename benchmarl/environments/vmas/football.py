#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING

    # Agents config
    n_blue_agents: int = MISSING
    n_red_agents: int = MISSING
    ai_red_agents: bool = MISSING
    physically_different: bool = MISSING

    # Agent spawning
    spawn_in_formation: bool = MISSING
    formation_agents_per_column: int = MISSING
    randomise_formation_indices: bool = MISSING
    only_blue_formation: bool = MISSING
    formation_noise: float = MISSING

    # Opponent heuristic config
    n_traj_points: int = MISSING
    ai_strength: float = MISSING
    ai_decision_strength: float = MISSING
    ai_precision_strength: float = MISSING

    # Task sizes
    agent_size: float = MISSING
    goal_size: float = MISSING
    goal_depth: float = MISSING
    pitch_length: float = MISSING
    pitch_width: float = MISSING
    ball_mass: float = MISSING
    ball_size: float = MISSING

    # Actions
    u_multiplier: float = MISSING

    # Actions shooting
    enable_shooting: bool = MISSING
    u_rot_multiplier: float = MISSING
    u_shoot_multiplier: float = MISSING
    shooting_radius: float = MISSING
    shooting_angle: float = MISSING

    # Speeds
    max_speed: float = MISSING
    ball_max_speed: float = MISSING

    # Rewards
    dense_reward: bool = MISSING
    pos_shaping_factor_ball_goal: float = MISSING
    pos_shaping_factor_agent_ball: float = MISSING
    distance_to_ball_trigger: float = MISSING
    scoring_reward: float = MISSING

    # Observations
    observe_teammates: bool = MISSING
    observe_adversaries: bool = MISSING
    dict_obs: bool = MISSING
