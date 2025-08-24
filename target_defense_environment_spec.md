# Target Defense Environment Specification

## Overview
This document describes the Target Defense environment for debugging and validation purposes.

## Environment Configuration
```yaml
# From benchmarl/conf/task/vmas/target_defense.yaml
max_steps: 100
num_defenders: 3
num_attackers: 1
sensing_radius: 0.1
attacker_sensing_radius: 0.1
speed_ratio: 0.1
target_distance: 0.05
randomize_attacker_x: true
num_spawn_positions: 3
enable_wall_constraints: true
wall_epsilon: 0.03
fixed_attacker_policy: true
```

## Agent Groups
- **Defenders**: 3 agents (controllable by RL)
- **Attackers**: 1 agent (fixed policy or controllable)
- **Total Agents**: 4

## Environment Space
- **World Size**: 2.0 x 2.0 (from -1.0 to +1.0 in both dimensions)
- **Target Area**: Located at x=1.0 (right edge)
- **Spawn Areas**: Left side of environment

## Action Space
- **Type**: Continuous actions (2D)
- **Format**: `[heading_angle, unused]`
- **Heading Range**: [-1, 1] (normalized) → [-π, π] radians
  - -1.0 → -π (180° West)
  - 0.0 → 0 (0° East)  
  - 1.0 → π (180° West)
- **Speed**: Fixed at maximum speed (agents always move at max velocity)
- **Second Dimension**: Ignored (required by VMAS but unused)

## Observation Space
**Per Agent Observations:**
1. **Own Position**: (x, y) coordinates
2. **Own Velocity**: (vx, vy) velocity vector
3. **Target Information**: Distance and direction to target
4. **Other Agents**: Position and velocity of sensed agents within radius
5. **Wall Constraints**: Distance to walls if enabled

**Expected Observation Size**: Variable based on number of sensed agents

## Reward Structure
**Defenders:**
- **Positive**: For sensing/intercepting attackers
- **Negative**: When attackers reach target

**Attackers:**
- **Positive**: For reaching target undetected
- **Negative**: For being sensed/intercepted

## Key Environment Events
1. **Sensing**: When defender is within `sensing_radius` of attacker
2. **Interception**: When defender touches attacker (distance < threshold)
3. **Target Reached**: When attacker reaches x ≥ (1.0 - target_distance)
4. **Wall Collision**: When agent hits environment boundaries

## Custom Metrics Logged
```python
info_dict = {
    "attackers_sensed": int,          # Number of attackers currently sensed
    "attackers_intercepted": int,     # Number of attackers intercepted this episode
    "attackers_reached_target": int,  # Number of attackers that reached target
    "sensing_occurred": bool,         # True if any sensing happened this step
    "interception_occurred": bool,    # True if any interception happened this step
    "target_reached": bool,           # True if any attacker reached target this step
    "attacker_rewards": float         # Total attacker rewards this step
}
```

## Environment Lifecycle
1. **Reset**: Agents spawn in random positions (defenders left, attackers configurable)
2. **Step**: 
   - Process actions → update positions → check events → calculate rewards
   - Wall constraints applied if enabled
   - Fixed attacker policy executed if enabled
3. **Done**: Episode ends when max_steps reached or all attackers intercepted/reached target

## Potential Issues Identified

### 1. Zero Rewards Problem
**Symptoms**: All reward metrics showing 0 in wandb logs
**Possible Causes**:
- Agents not moving (action processing issue)
- Sensing radius too small for interactions
- Reward calculation bugs
- Environment reset issues

### 2. NaN Training Values
**Symptoms**: Defender training shows NaN for ESS, entropy, kl_approx
**Possible Causes**:
- Action explosion (gradients becoming infinite)
- Division by zero in reward calculations
- Observation normalization issues

### 3. Performance Issues
**Symptoms**: 25+ seconds per iteration
**Possible Causes**:
- Complex reward calculations
- Inefficient sensing computations
- Wall constraint calculations
- Missing apollonius solver dependency

## Debugging Checklist

### Basic Functionality
- [ ] Agents spawn in correct positions
- [ ] Actions are processed correctly (heading conversion)
- [ ] Agents actually move each step
- [ ] Sensing detection works within radius
- [ ] Target detection works at boundaries
- [ ] Rewards are non-zero when events occur

### Action Processing
- [ ] Action shape is correct: (batch_size, num_agents, 2)
- [ ] Heading conversion: [-1,1] → [-π,π] works
- [ ] No NaN values in processed actions
- [ ] Wall constraints don't produce invalid actions

### Observation Processing  
- [ ] Observation shapes match expected sizes
- [ ] All observation values are finite (no NaN/inf)
- [ ] Sensing-based observations update correctly
- [ ] Position/velocity values are reasonable

### Reward Calculation
- [ ] Rewards are calculated when events occur
- [ ] Reward values are finite and reasonable
- [ ] Info dict metrics update correctly
- [ ] Episode termination works properly

### Performance Optimization
- [ ] Vectorized operations used where possible
- [ ] Minimal redundant calculations
- [ ] Efficient distance computations
- [ ] Optional features (like apollonius solver) don't block execution

## Validation Commands

### Test Basic Environment
```bash
python -c "
import torch
from vmas import make_env
env = make_env(scenario='target_defense', num_envs=1, device='cpu', continuous_actions=True)
obs = env.reset()
print('Obs shape:', {k: v.shape for k, v in obs.items()})
action = torch.zeros((1, env.n_agents, 2))
obs, reward, done, info = env.step(action)
print('Reward:', reward)
print('Info:', info)
"
```

### Test Training Speed
```bash
python -m benchmarl.run task=vmas/target_defense algorithm=mappo experiment=base_experiment seed=0 \
  experiment.max_n_frames=10000 \
  experiment.evaluation=false \
  experiment.render=false
```

### Compare with Working Environment
```bash
python -m benchmarl.run task=vmas/simple_tag algorithm=mappo experiment=base_experiment seed=0 \
  experiment.max_n_frames=10000 \
  experiment.evaluation=false \
  experiment.render=false
```

## Expected Behavior

### Normal Operation
- Agents should move around the environment
- Defenders should learn to intercept attackers
- Attackers should learn to avoid defenders (if not fixed policy)
- Rewards should be non-zero and meaningful
- Training metrics should be stable (no NaN values)

### Performance Targets
- <5 seconds per iteration with optimized settings
- Non-zero reward progression over time
- Stable training without NaN explosions
- Meaningful learning curves in wandb

## Next Steps for Debugging
1. Run the validation commands above
2. Check if basic environment functionality works
3. Verify action/observation processing
4. Test reward calculations manually
5. Profile performance bottlenecks
6. Compare behavior with simple_tag environment