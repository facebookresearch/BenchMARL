# Target Defense Environment - Detailed Algorithmic Breakdown

## 1. ENVIRONMENT INITIALIZATION (`make_world`)

### 1.1 Parameter Setup
```
Input: batch_dim, device, **kwargs
- Extract: num_defenders=3, num_attackers=1, sensing_radius=0.15, speed_ratio=0.2
- Set speeds: defender_speed=0.05, attacker_speed=0.05*0.2=0.01
- Create VMAS world: x_semidim=0.5, y_semidim=0.5 (internal [-0.5,0.5])
```

### 1.2 Agent Creation
```
For each defender (3 agents):
  - Create Agent with max_speed=0.05, color=blue, is_defender=True
  - Add to world

For each attacker (1 agent):
  - Create Agent with max_speed=0.01, color=red, is_defender=False  
  - Add to world
```

### 1.3 Tracking Variables Initialization
```
Initialize tracking tensors (batch_dim x num_attackers):
- attacker_sensed: [False, False, ...] 
- attacker_intercepted: [False, False, ...]
- attacker_reached_target: [False, False, ...]
- attacker_sensing_rewards: [0.0, 0.0, ...]
- defender_has_sensed: [False, False, ...] (batch_dim x num_defenders)
- step_count: [0, 0, ...] (batch_dim)
```

---

## 2. EPISODE RESET (`reset_world_at`)

### 2.1 Coordinate System Setup
```
World coordinates [0,1] ←→ VMAS coordinates [-0.5,0.5]
Conversion functions:
- world_to_vmas(coord) = coord - 0.5
- vmas_to_world(coord) = coord + 0.5
```

### 2.2 Agent Positioning
```
DEFENDERS (Bottom Edge Defense Line):
- World Y = 0.0 (bottom edge/target line)
- World X = [0.25, 0.5, 0.75] (evenly spaced)
- Convert to VMAS: positions = [(-0.25,-0.5), (0.0,-0.5), (0.25,-0.5)]

ATTACKERS (Top Edge Spawn):
- World Y = 1.0 (top edge/spawn line) 
- World X = randomized across [0.1, 0.9] with num_spawn_positions=3
- Convert to VMAS: position = (random_x-0.5, 0.5)
```

### 2.3 Reset Tracking Variables
```
Reset all tracking tensors to initial state:
- All sensed flags → False
- All rewards → 0.0  
- Step count → 0
- Event flags → False
```

---

## 3. STEP EXECUTION

### 3.1 Action Processing (`process_action`) - Per Agent

#### 3.1.A Event Flag Reset
```
IF _reset_events_flag_next_step == True:
    _events_updated_this_step = False
    _step_incremented_this_step = False
    _reset_events_flag_next_step = False
```

#### 3.1.B Action Interpretation
```
INPUT: action.u[:, 0] ∈ [-1, 1] (normalized heading)
CONVERT: heading = action.u[:, 0] * π  (to [-π, π] radians)
CONVERT: theta = heading % (2π)  (to [0, 2π])

FOR DEFENDERS:
  IF enable_wall_constraints:
    theta = apply_wall_constraints(agent, theta)
  
FOR ATTACKERS:
  IF fixed_attacker_policy:
    theta = -π/2  (always move down toward target)
  ELSE:
    theta = use action heading
```

#### 3.1.C Speed Determination
```
FOR DEFENDERS:
  IF defender_has_sensed[defender_idx] == True:
    max_speed = 0  (defender becomes inactive after sensing)
  ELSE:
    max_speed = 0.05

FOR ATTACKERS:  
  IF attacker_sensed[attacker_idx] OR attacker_reached_target[attacker_idx]:
    max_speed = 0  (attacker becomes inactive)
  ELSE:
    max_speed = 0.01
```

#### 3.1.D Velocity Computation
```
action.u[:, 0] = max_speed * cos(theta)
action.u[:, 1] = max_speed * sin(theta)
```

#### 3.1.E Event Updates (Called Once Per Step)
```
CALL: update_events()  (updates sensing, interception, target reach)
INCREMENT: step_count += 1  (once per step)
```

---

## 4. EVENT DETECTION (`update_events`)

### 4.1 Sensing Detection
```
FOR each attacker_idx, attacker:
  FOR each defender_idx, defender:
    
    # Calculate distance
    dist = ||attacker.pos - defender.pos||
    
    # Check sensing condition  
    newly_sensed = (dist ≤ sensing_radius) AND NOT attacker_sensed[env_idx, attacker_idx]
    
    IF newly_sensed:
      # Mark events
      defender_has_sensed[env_idx, defender_idx] = True
      attacker_sensed[env_idx, attacker_idx] = True
      
      # Snap attacker to sensing boundary
      direction = (attacker.pos - defender.pos) / ||attacker.pos - defender.pos||
      attacker.pos = defender.pos + direction * sensing_radius
      attacker.vel = 0
      
      # Compute reward using Apollonius solver
      CALL: compute_apollonius_reward(env_idx, attacker_idx)
```

### 4.2 Apollonius Reward Computation
```
# Convert positions to global frame [0,1]
attacker_pos_world = vmas_to_world(attacker.pos)
defender_positions_world = [vmas_to_world(def.pos) for def in defenders]

# Call Apollonius optimization
result = solve_apollonius_optimization(
  attacker_pos=attacker_pos_world,
  defender_positions=defender_positions_world,  
  nu = defender_speed / attacker_speed = 0.05 / 0.01 = 5.0
)

IF result['success']:
  # Use pure Apollonius defender payoff
  attacker_sensing_rewards[env_idx, attacker_idx] = result['defender_payoff']
  
  IF result['defender_payoff'] > 0:
    attacker_intercepted[env_idx, attacker_idx] = True
    
ELSE:
  # Fallback: y-position in global frame
  attacker_y_world = vmas_to_world(attacker.pos[Y])
  attacker_sensing_rewards[env_idx, attacker_idx] = attacker_y_world
```

### 4.3 Target Reach Detection
```
target_y_vmas = world_to_vmas(0.0) = -0.5  # Bottom edge in VMAS

FOR each attacker_idx, attacker:
  reached = (attacker.pos[Y] ≤ target_y_vmas + target_distance) 
            AND NOT attacker_sensed[env_idx, attacker_idx]
  
  attacker_reached_target[env_idx, attacker_idx] |= reached
```

---

## 5. REWARD CALCULATION (`reward`) - Per Agent

### 5.1 Terminal Reward Structure
```
INPUT: agent (defender or attacker)
OUTPUT: reward tensor

IF agent.is_defender:
  done_mask = check_episode_done()
  
  IF any episode is done:
    FOR each completed env_idx:
      total_reward = 0.0
      
      FOR each attacker:
        IF attacker_sensed[env_idx, attacker_idx]:
          total_reward += attacker_sensing_rewards[env_idx, attacker_idx]
      
      # Share reward equally among all defenders
      shared_reward = total_reward / num_defenders
      reward[env_idx] = shared_reward

ELSE: # Attacker
  reward = 0.0  (attackers get no rewards)
```

---

## 6. EPISODE TERMINATION (`done`)

### 6.1 Termination Conditions
```
Episode ends when ANY of:

1. ALL attackers sensed OR reached target:
   all_attackers_done = (attacker_sensed | attacker_reached_target).all(dim=1)

2. Maximum steps reached:
   max_steps_reached = (step_count ≥ max_steps)

done = all_attackers_done | max_steps_reached
```

---

## 7. OBSERVATION STRUCTURE (`observation`)

### 7.1 Observation Composition
```
Total observation size: (num_defenders + num_attackers) * 2 = 8

Observation order: [def0_x, def0_y, def1_x, def1_y, def2_x, def2_y, att0_x, att0_y]

FOR each other_agent:
  IF other_agent == self:
    # Always observe own position
    obs[idx:idx+2] = other_agent.pos
    
  ELIF same_team(agent, other_agent):
    # Same team always visible  
    obs[idx:idx+2] = other_agent.pos
    
  ELSE: # Opponent
    # Only visible if within sensing radius
    dist = ||agent.pos - other_agent.pos||
    IF dist ≤ agent.sensing_radius:
      obs[idx:idx+2] = other_agent.pos
    ELSE:
      obs[idx:idx+2] = [0, 0]  # Unobserved
```

---

## 8. COORDINATE SYSTEM DETAILS

### 8.1 Coordinate Transformations
```
VMAS Internal: [-0.5, 0.5] × [-0.5, 0.5]
World Logical: [0, 1] × [0, 1]

Key positions:
- Target line: y=0 (world) = y=-0.5 (VMAS)  
- Spawn line: y=1 (world) = y=0.5 (VMAS)
- Defender start: y=0, x∈[0.25, 0.5, 0.75] (world)
- Attacker start: y=1, x∈random (world)
```

### 8.2 Speed and Movement
```
Time to traverse world (1 unit):
- Defenders: 1.0 / 0.05 = 20 steps
- Attackers: 1.0 / 0.01 = 100 steps  

Attacker time to target: ~100 steps
Episode max_steps: 250 steps
Defender advantage: 5x faster (speed_ratio=0.2)
```

---

## 9. CURRENT CONFIGURATION

```yaml
Environment Parameters:
- World: [0,1] × [0,1] (logical), [-0.5,0.5] × [-0.5,0.5] (VMAS)
- Agents: 3 defenders + 1 attacker
- Speeds: defender=0.05, attacker=0.01 (ratio=0.2)
- Sensing: radius=0.15
- Episode: max_steps=250
- Rewards: Pure terminal, Apollonius-based, shared equally
- Constraints: Wall constraints disabled
```

## 10. REWARD FLOW SUMMARY

```
1. Agents move → positions update
2. Distance calculation → sensing detection  
3. IF sensing occurs → Apollonius solver called with global [0,1] coordinates
4. Apollonius returns defender_payoff → stored as sensing reward
5. At episode end → total rewards shared equally among all defenders
6. Attackers always get 0 reward
```

This is the complete algorithmic process. Each step follows this exact sequence every timestep.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create detailed algorithmic breakdown of environment process", "status": "completed"}, {"content": "Document step-by-step execution flow", "status": "completed"}, {"content": "Verify each component is working correctly", "status": "completed"}]