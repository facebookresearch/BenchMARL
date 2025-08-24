# Target Defense Environment - Complete Specification

## Environment Overview

**Scenario Type**: Multi-agent cooperative-competitive  
**Framework**: VMAS (Vectorized Multi-Agent Simulator)  
**Task**: Defenders protect a target line from attacking agents  
**Learning Type**: Reinforcement Learning with continuous actions  

## World Configuration

### Physical World
- **Coordinate System**: Cartesian 2D
- **World Bounds**: x ∈ [-0.5, 0.5], y ∈ [-0.5, 0.5] (1.0 × 1.0 square)
- **Target Line**: Located at x = 0.5 (right boundary)
- **Spawn Area**: Defenders at y = -0.5 (bottom), Attackers at y = 0.5 (top)
- **Physics**: No collisions between agents (collision_force = 0)
- **Time Step**: dt = 1.0 (unit timestep)
- **Substeps**: 1 (no sub-physics simulation)

### Agent Specifications

#### Defenders
- **Count**: 3 (configurable via `num_defenders`)
- **Role**: Sense and intercept attackers before they reach target
- **Shape**: Sphere with radius 0.02
- **Color**: Blue (0.0, 0.0, 1.0)
- **Max Speed**: 0.05 units/step
- **Sensing Radius**: 0.1 (configurable via `sensing_radius`)
- **Spawn Pattern**: Evenly distributed along y = -0.5
- **Control**: RL-controlled actions

#### Attackers  
- **Count**: 1 (configurable via `num_attackers`)
- **Role**: Reach target line while avoiding detection
- **Shape**: Sphere with radius 0.02
- **Color**: Red (1.0, 0.0, 0.0)
- **Max Speed**: 0.05 × speed_ratio (default: 0.035)
- **Sensing Radius**: 0.1 (configurable via `attacker_sensing_radius`)
- **Spawn Pattern**: Center at (0.0, 0.5) or randomized positions
- **Control**: Fixed policy or RL-controlled

### Configuration Parameters

```python
# From YAML configuration
max_steps: 100                    # Episode length
num_defenders: 3                  # Number of defending agents
num_attackers: 1                  # Number of attacking agents
sensing_radius: 0.1               # Defender sensing range
attacker_sensing_radius: 0.1      # Attacker sensing range
speed_ratio: 0.1                  # Attacker speed = defender_speed × ratio
target_distance: 0.05             # Distance threshold for reaching target
randomize_attacker_x: true        # Randomize attacker spawn positions
num_spawn_positions: 3            # Number of possible spawn locations
enable_wall_constraints: true     # Apply wall collision constraints
wall_epsilon: 0.03                # Distance from wall to start constraining
fixed_attacker_policy: true       # Use fixed policy for attackers
```

## Action Space

### Action Format
- **Type**: Continuous
- **Dimensions**: 2 per agent
- **Shape**: `(batch_size, num_agents, 2)`
- **Action Vector**: `[heading_angle, unused]`

### Heading Control
- **Input Range**: [-1.0, 1.0] (normalized)
- **Mapping**: Linear to [-π, π] radians
- **Conversion**: `theta = action[0] * π`
- **Direction Examples**:
  - -1.0 → -π radians (-180°, West)
  - -0.5 → -π/2 radians (-90°, South)
  - 0.0 → 0 radians (0°, East)
  - 0.5 → π/2 radians (90°, North)
  - 1.0 → π radians (180°, West)

### Movement Model
- **Speed**: Fixed at agent's max_speed
- **Velocity Calculation**: `velocity = [cos(theta), sin(theta)] × max_speed`
- **Position Update**: `new_pos = old_pos + velocity × dt`
- **Second Action Dimension**: Ignored (required by VMAS interface)

### Wall Constraints
When `enable_wall_constraints = true`:
- **Trigger Distance**: `wall_epsilon = 0.03` from boundaries
- **Constraint Method**: Heading angle clamping to valid directions
- **Wall Regions**:
  - **Right Wall** (x > 0.5 - ε): Clamp to [π/2, 3π/2]
  - **Left Wall** (x < -0.5 + ε): Clamp to [0, π/2] ∪ [3π/2, 2π]
  - **Top Wall** (y > 0.5 - ε): Clamp to [π, 2π]
  - **Bottom Wall** (y < -0.5 + ε): Clamp to [0, π]
  - **Corners**: Intersection of adjacent wall constraints

## Observation Space

### Per-Agent Observations
**Shape**: `(batch_size, obs_size)`  
**Type**: Continuous floating-point values  
**Base Size**: 8 dimensions per agent  

### Observation Components
1. **Own State** (4 dimensions):
   - `pos_x`: Agent's x-coordinate ∈ [-0.5, 0.5]
   - `pos_y`: Agent's y-coordinate ∈ [-0.5, 0.5]
   - `vel_x`: Agent's x-velocity ∈ [-max_speed, max_speed]
   - `vel_y`: Agent's y-velocity ∈ [-max_speed, max_speed]

2. **Target Information** (2 dimensions):
   - `target_distance`: Euclidean distance to target line
   - `target_direction`: Angle to target relative to current heading

3. **Sensing Information** (2+ dimensions):
   - **Base**: 2 dimensions for "no agents sensed" case
   - **Variable**: Additional dimensions for each sensed agent
   - **Per Sensed Agent**: Position and velocity relative to observer

### Sensing Mechanism
- **Detection Range**: Agent-specific sensing radius
- **Condition**: `distance(agent_i, agent_j) ≤ sensing_radius_i`
- **Information**: Position and velocity of sensed agents
- **Privacy**: Agents only observe others within their sensing range

## State Tracking

### Episode State Variables
```python
# Shape: (batch_dim, num_attackers)
attacker_sensed: torch.Tensor           # Boolean - currently being sensed
attacker_intercepted: torch.Tensor      # Boolean - intercepted this episode
attacker_reached_target: torch.Tensor   # Boolean - reached target this episode
attacker_sensing_rewards: torch.Tensor  # Float - accumulated sensing rewards

# Shape: (batch_dim, num_defenders)  
defender_has_sensed: torch.Tensor       # Boolean - has sensed any attacker
```

### Event Detection
1. **Sensing Event**: Defender within sensing_radius of attacker
2. **Interception Event**: Using Apollonius circle optimization (if available)
3. **Target Reached**: Attacker x-position ≥ (0.5 - target_distance)
4. **Wall Collision**: Agent position within wall_epsilon of boundary

### Update Cycle
- **Frequency**: Every simulation step
- **Order**: Position update → Event detection → Reward calculation
- **Persistence**: Events tracked across episode duration

## Reward Structure

### Reward Calculation Timing
- **Type**: Sparse rewards at episode termination
- **Condition**: Only when `done() = True`
- **Intermediate**: No step-by-step rewards during episode

### Defender Rewards
```python
def reward(self, agent: Agent) -> torch.Tensor:
    if agent.is_defender and done_mask.any():
        for env_idx in torch.where(done_mask)[0]:
            total = 0.0
            for a_idx in range(self.num_attackers):
                if self.attacker_sensed[env_idx, a_idx]:
                    total += self.attacker_sensing_rewards[env_idx, a_idx]
            r[env_idx] = total
    return r
```

### Reward Components
- **Sensing Rewards**: Based on Apollonius circle optimization (if available)
- **Interception Bonus**: Additional reward for successful interceptions
- **Target Defense**: Penalty if attackers reach target
- **Fallback**: Simple distance-based rewards if Apollonius unavailable

### Apollonius Solver Integration
- **Purpose**: Optimal interception point calculation
- **Availability**: Optional dependency (`apollonius_solver`)
- **Fallback**: Basic geometric rewards if unavailable
- **Warning**: "Warning: apollonius_solver not available. Using fallback rewards."

## Episode Termination

### Done Conditions
```python
def done(self) -> torch.Tensor:
    if not hasattr(self, 'attacker_sensed') or self.attacker_sensed is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    # Additional termination logic...
```

### Termination Triggers
1. **Max Steps**: Episode length reaches `max_steps = 100`
2. **All Intercepted**: All attackers have been intercepted
3. **Target Reached**: Any attacker reaches the target line
4. **Early Termination**: Based on scenario-specific conditions

## Information Dictionary

### Custom Metrics (Per Step)
```python
info = {
    "attackers_sensed": int,           # Current number of sensed attackers
    "attackers_intercepted": int,      # Total intercepted this episode  
    "attackers_reached_target": int,   # Total reached target this episode
    "sensing_occurred": bool,          # Any sensing this step
    "interception_occurred": bool,     # Any interception this step
    "target_reached": bool,            # Any target reached this step
    "attacker_rewards": float          # Total attacker rewards this step
}
```

### Logged Metrics in BenchMARL
- **Collection Phase**: Real-time metrics during training
- **Evaluation Phase**: Performance metrics during testing
- **Agent Groups**: Separate metrics for attackers and defenders
- **Aggregation**: Min, max, mean values across episodes

## Performance Characteristics

### Computational Complexity
- **Agent Updates**: O(num_agents) per step
- **Sensing Detection**: O(num_agents²) pairwise distance checks
- **Wall Constraints**: O(num_agents) boundary checks
- **Reward Calculation**: O(num_attackers × num_defenders)

### Memory Usage
- **State Tensors**: (batch_dim, num_agents, state_size)
- **Tracking Arrays**: (batch_dim, num_attackers) for events
- **Observation Buffers**: Variable size based on sensing

### Training Performance Issues
1. **Sparse Rewards**: Long episodes (100 steps) with rewards only at end
2. **Zero Feedback**: No intermediate learning signals
3. **Numerical Instability**: NaN values in defender training metrics
4. **Slow Iterations**: 25+ seconds per training iteration

## Integration with BenchMARL

### Task Registration
- **Location**: `benchmarl/environments/vmas/common.py`
- **Entry**: `TARGET_DEFENSE = None` in `VmasTask` enum
- **Configuration**: `benchmarl/conf/task/vmas/target_defense.yaml`

### Environment Loading
- **Path**: `/opt/anaconda3/envs/benchmarl2/lib/python3.10/site-packages/vmas/scenarios/target_defense.py`
- **Class**: `Scenario(BaseScenario)`
- **Import**: Automatic via VMAS scenario loading system

### Training Configuration
```yaml
# Optimized settings for performance
experiment:
  on_policy_n_envs_per_worker: 64     # Parallel environments
  parallel_collection: true           # Multi-core data collection
  on_policy_collected_frames_per_batch: 12000  # Larger batches
  evaluation: false                    # Disable during development
  render: false                        # Disable rendering
```

## Known Issues and Limitations

### Critical Issues
1. **Reward Sparsity**: No intermediate rewards, only end-of-episode
2. **Training Instability**: NaN values in defender training
3. **Performance**: Slow training iterations (25s+ per iteration)
4. **Zero Metrics**: All custom metrics showing 0 values

### Dependencies
- **Required**: `torch`, `vmas`, `numpy`
- **Optional**: `apollonius_solver` (performance optimization)
- **Development**: `benchmarl`, `wandb`, `moviepy`

### Environment Limitations
- **Fixed Episode Length**: Always 100 steps regardless of events
- **Sparse Feedback**: Agents receive no learning signals during episodes
- **Simple Physics**: No complex dynamics or realistic motion models
- **Limited Scalability**: Performance degrades with many agents

### BenchMARL Integration Issues
- **Rendering Crashes**: `extra_render()` method compatibility
- **Action Format**: Mismatch between expected and provided action shapes
- **Configuration**: Parameter passing between YAML and environment
- **Metrics Logging**: Custom info dict integration with wandb

## Debugging and Validation

### Environment Testing
```python
# Basic functionality test
import torch
from vmas import make_env
env = make_env(scenario='target_defense', num_envs=1, device='cpu', continuous_actions=True)
obs = env.reset()
actions = [torch.tensor([[0.1, 0.0]]) for _ in range(env.n_agents)]
obs, reward, done, info = env.step(actions)
```

### Performance Profiling
- **Step Timing**: Measure individual step execution time
- **Memory Usage**: Monitor tensor allocation and deallocation
- **GPU Utilization**: Check device usage during training
- **I/O Bottlenecks**: Identify file system or network delays

### Validation Metrics
- **Agent Movement**: Verify position updates after actions
- **Event Detection**: Confirm sensing and interception logic
- **Reward Calculation**: Test reward assignment under various scenarios
- **Episode Termination**: Validate done conditions and reset behavior