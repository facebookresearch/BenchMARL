#!/usr/bin/env python3

"""Test that environment passes correct global coordinates to Apollonius solver"""

import torch
import vmas
import numpy as np
import math

# Test if Apollonius solver is available
try:
    from apollonius_solver import solve_apollonius_optimization
    APOLLONIUS_AVAILABLE = True
    print("✅ Apollonius solver available")
except ImportError:
    APOLLONIUS_AVAILABLE = False
    print("❌ Apollonius solver not available")

def test_coordinate_passing():
    print("=== Testing Apollonius Coordinate Passing ===")
    
    # Import our environment
    from benchmarl.environments.vmas.target_defense import Scenario
    
    # Create environment
    scenario = Scenario()
    env = vmas.make_env(
        scenario=scenario,
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        max_steps=250,
        speed_ratio=0.2,
        sensing_radius=0.15,
        num_defenders=3,
        num_attackers=1,
        use_apollonius=True
    )
    
    print(f"Apollonius enabled: {scenario.use_apollonius}")
    print(f"Speed ratio: {scenario.speed_ratio}")
    
    # Reset environment
    obs = env.reset()
    
    # Print initial positions in both coordinate systems
    print("\n=== Initial Positions ===")
    for i, agent in enumerate(env.agents):
        vmas_pos = agent.state.pos[0].numpy()
        world_pos = scenario._vmas_to_world(vmas_pos)
        agent_type = "Defender" if agent.is_defender else "Attacker"
        print(f"{agent_type} {agent.name}:")
        print(f"  VMAS coords: {vmas_pos}")
        print(f"  World coords [0,1]: {world_pos}")
    
    # Add temporary debug to Apollonius call
    import sys
    sys.path.insert(0, '/Users/goutamdas/Library/CloudStorage/OneDrive-GeorgeMasonUniversity-O365Production/Research/Thesis_work/game3v1/benchCode/new_bench/BenchMARL')
    
    # Manual Apollonius test with current positions
    if APOLLONIUS_AVAILABLE:
        print("\n=== Manual Apollonius Test ===")
        
        # Get positions
        defenders = [a for a in env.agents if a.is_defender]
        attackers = [a for a in env.agents if not a.is_defender]
        
        attacker = attackers[0]
        attacker_pos_vmas = attacker.state.pos[0].cpu().numpy()
        attacker_pos_world = scenario._vmas_to_world(attacker_pos_vmas)
        
        defender_positions_world = []
        for defender in defenders:
            def_pos_vmas = defender.state.pos[0].cpu().numpy()
            def_pos_world = scenario._vmas_to_world(def_pos_vmas)
            defender_positions_world.append(def_pos_world)
        
        print(f"Attacker position (world): {attacker_pos_world}")
        print(f"Defender positions (world): {defender_positions_world}")
        
        # Call Apollonius solver directly
        result = solve_apollonius_optimization(
            attacker_pos=attacker_pos_world,
            defender_positions=defender_positions_world,
            nu=1.0 / scenario.speed_ratio  # nu = defender_speed / attacker_speed
        )
        
        print(f"\nApollonius result: {result}")
        
        # Verify coordinates are in [0,1] range
        print(f"\nCoordinate validation:")
        print(f"Attacker x in [0,1]: {0 <= attacker_pos_world[0] <= 1}")
        print(f"Attacker y in [0,1]: {0 <= attacker_pos_world[1] <= 1}")
        for i, def_pos in enumerate(defender_positions_world):
            print(f"Defender {i} x in [0,1]: {0 <= def_pos[0] <= 1}")
            print(f"Defender {i} y in [0,1]: {0 <= def_pos[1] <= 1}")
    
    # Test sensing scenario by moving defenders toward attacker
    print("\n=== Testing Sensing Scenario ===")
    
    # Create actions to make defenders move toward attacker
    for step in range(30):
        attacker_pos = None
        for agent in env.agents:
            if not agent.is_defender:
                attacker_pos = agent.state.pos[0]
                break
        
        actions = []
        for agent in env.agents:
            if agent.is_defender:
                # Move toward attacker
                if attacker_pos is not None:
                    def_pos = agent.state.pos[0]
                    direction = attacker_pos - def_pos
                    angle = torch.atan2(direction[1], direction[0])
                    normalized_angle = angle / math.pi
                    action = torch.tensor([[normalized_angle.item(), 0.0]])
                else:
                    action = torch.tensor([[0.5, 0.0]])  # Move up
            else:
                action = torch.tensor([[-0.5, 0.0]])  # Move down
            actions.append(action)
        
        obs, rewards, dones, info = env.step(actions)
        
        # Check for sensing events
        step_rewards = [r[0].item() for r in rewards]
        max_reward = max(step_rewards)
        
        if max_reward > 0:
            print(f"\nStep {step}: SENSING OCCURRED!")
            print(f"Rewards: {step_rewards}")
            
            # Check positions passed to solver
            if hasattr(scenario, 'attacker_sensing_rewards'):
                sensing_reward = scenario.attacker_sensing_rewards[0, 0].item()
                print(f"Sensing reward value: {sensing_reward}")
                
                # Check if this looks like Apollonius (typically 0.1-0.8) or y-position (0.0-1.0)
                if 0.1 <= sensing_reward <= 0.8:
                    print("✅ Looks like Apollonius payoff")
                elif 0.0 <= sensing_reward <= 1.0:
                    print("⚠️ Looks like y-position fallback")
                else:
                    print("❓ Unusual reward value")
            break
        
        if dones[0]:
            print(f"Episode ended at step {step}")
            break
    else:
        print("No sensing occurred in 30 steps")

if __name__ == "__main__":
    test_coordinate_passing()