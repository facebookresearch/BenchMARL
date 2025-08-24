#!/usr/bin/env python3

"""Test the exact environment being used by BenchMARL"""

import torch
import vmas
from benchmarl.environments.vmas.target_defense import Scenario
import math
import numpy as np

def test_environment():
    print("=== Testing Current BenchMARL Environment ===")
    
    # Create environment with exact same parameters as training
    scenario = Scenario()
    env = vmas.make_env(
        scenario=scenario,
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        # Use the exact parameters from the YAML config
        max_steps=250,
        num_defenders=3,
        num_attackers=1,
        sensing_radius=0.15,
        speed_ratio=0.2,
        randomize_attacker_x=True,
        num_spawn_positions=3,
        fixed_attacker_policy=True
    )
    
    print(f"Environment parameters:")
    print(f"  Speed ratio: {scenario.speed_ratio}")
    print(f"  Sensing radius: {scenario.sensing_radius}")
    print(f"  Max steps: {scenario.max_steps}")
    print(f"  Defender speed: {scenario.defender_max_speed}")
    print(f"  Attacker speed: {scenario.attacker_max_speed}")
    
    # Reset environment
    obs = env.reset()
    print(f"\nInitial positions:")
    for i, agent in enumerate(env.agents):
        vmas_pos = agent.state.pos[0].numpy()
        world_pos = scenario._vmas_to_world(vmas_pos)
        agent_type = "Defender" if agent.is_defender else "Attacker"
        print(f"  {agent_type} {agent.name}: World {world_pos}")
    
    # Test with strategic actions to ensure sensing
    print(f"\n=== Testing Sensing ===")
    for step in range(30):
        # Get attacker position
        attacker_pos = None
        for agent in env.agents:
            if not agent.is_defender:
                attacker_pos = agent.state.pos[0]
                break
        
        # Create actions for all agents
        actions = []
        for agent in env.agents:
            if agent.is_defender:
                # Move defenders towards attacker
                if attacker_pos is not None:
                    def_pos = agent.state.pos[0]
                    direction = attacker_pos - def_pos
                    angle = torch.atan2(direction[1], direction[0])
                    normalized_angle = angle / math.pi
                    action = torch.tensor([[normalized_angle.item(), 0.0]])
                else:
                    action = torch.tensor([[0.5, 0.0]])  # Move up
            else:
                # Attacker action (will be overridden by fixed policy)
                action = torch.tensor([[-0.5, 0.0]])  # Move down
            actions.append(action)
        
        # Step environment
        obs, rewards, dones, info = env.step(actions)
        
        # Check rewards and distances
        step_rewards = [r[0].item() for r in rewards]
        max_reward = max(step_rewards)
        
        if max_reward > 0:
            print(f"Step {step}: SUCCESS! Rewards: {step_rewards}")
            print(f"  Episode done: {dones[0].item()}")
            break
        
        # Check distances every few steps
        if step % 5 == 0:
            distances = []
            for i, def_agent in enumerate(env.agents[:3]):  # First 3 are defenders
                for j, att_agent in enumerate(env.agents[3:]):  # Last is attacker
                    dist = torch.norm(def_agent.state.pos[0] - att_agent.state.pos[0]).item()
                    distances.append(f"Def{i}-Att{j}: {dist:.3f}")
            print(f"Step {step}: Distances: {', '.join(distances)}")
        
        if dones[0]:
            print(f"Step {step}: Episode ended")
            break
    
    print(f"\nFinal test result:")
    print(f"  Max reward achieved: {max(step_rewards)}")
    print(f"  Episode completed: {dones[0].item()}")
    
    # Check sensing flags
    if hasattr(scenario, 'attacker_sensed'):
        print(f"  Attacker sensed: {scenario.attacker_sensed[0, 0].item()}")
    if hasattr(scenario, 'attacker_reached_target'):
        print(f"  Attacker reached target: {scenario.attacker_reached_target[0, 0].item()}")

if __name__ == "__main__":
    test_environment()