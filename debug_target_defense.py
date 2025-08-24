#!/usr/bin/env python3

"""Debug script to test target defense environment"""

import torch
import numpy as np
import vmas
from benchmarl.environments.vmas.target_defense import Scenario

def debug_environment():
    print("=== Target Defense Environment Debug ===")
    
    # Create environment with current config parameters
    scenario = Scenario()
    env = vmas.make_env(
        scenario=scenario,
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        # Updated config parameters
        max_steps=200,
        num_defenders=3,
        num_attackers=1,
        sensing_radius=0.15,
        speed_ratio=0.1,  # Slow attackers for easier learning!
        randomize_attacker_x=True,
        num_spawn_positions=3,
        fixed_attacker_policy=True
    )
    
    print(f"Environment created with {env.n_agents} agents")
    print(f"Max steps: {scenario.max_steps}")
    print(f"Defender speed: {scenario.defender_max_speed}")
    print(f"Attacker speed: {scenario.attacker_max_speed}")
    print(f"Speed ratio: {scenario.speed_ratio}")
    print(f"Sensing radius: {scenario.sensing_radius}")
    
    # Reset and check initial positions
    obs = env.reset()
    print("\n=== Initial Positions (Global Frame [0,1]) ===")
    for i, agent in enumerate(env.agents):
        vmas_pos = agent.state.pos[0].numpy()
        global_pos = scenario._vmas_to_world(vmas_pos)
        agent_type = "Defender" if agent.is_defender else "Attacker"
        print(f"{agent_type} {agent.name}: Global {global_pos}")
    
    # Simulate some steps
    print("\n=== Simulation ===")
    total_steps = 0
    sensing_events = 0
    
    for step in range(20):  # Test first 20 steps
        # Smarter actions: defenders move towards attacker
        actions = []
        attacker_pos = None
        for agent in env.agents:
            if not agent.is_defender:
                attacker_pos = agent.state.pos[0]
                break
        
        for agent in env.agents:
            if agent.is_defender:
                # Move towards attacker
                if attacker_pos is not None:
                    def_pos = agent.state.pos[0]
                    direction = attacker_pos - def_pos
                    angle = torch.atan2(direction[1], direction[0])
                    # Normalize to [-1, 1] for action
                    normalized_angle = angle / np.pi
                    action = torch.tensor([[normalized_angle.item(), 0.0]])
                else:
                    action = torch.tensor([[0.5, 0.0]])  # Default up
            else:
                # Attacker moves down (policy overridden anyway)
                action = torch.tensor([[-0.5, 0.0]])  # -0.5 = -π/2 = down
            actions.append(action)
        
        obs, rewards, dones, info = env.step(actions)
        
        # Check for events
        step_rewards = [r[0].item() for r in rewards]
        step_max_reward = max(step_rewards)
        
        # Check if episode is done and why
        is_done = dones[0].item() if len(dones) > 0 else False
        if is_done:
            print(f"Step {step}: EPISODE DONE!")
            print(f"  Attacker sensed: {scenario.attacker_sensed[0, 0].item()}")
            print(f"  Attacker reached target: {scenario.attacker_reached_target[0, 0].item()}")
            print(f"  Step count: {scenario.step_count[0].item()}/{scenario.max_steps}")
        
        if step_max_reward > 0:
            sensing_events += 1
            print(f"Step {step}: SENSING EVENT! Rewards: {step_rewards}")
        
        # Check sensing status manually
        if step > 10:  # After some steps when they should be close
            print(f"Step {step}: Manual distance check")
            attacker = env.agents[3]  # Last agent is attacker
            for i, defender in enumerate(env.agents[:3]):
                dist = torch.norm(attacker.state.pos[0] - defender.state.pos[0]).item()
                sensed = hasattr(scenario, 'attacker_sensed') and scenario.attacker_sensed[0, 0].item()
                done = dones[0].item() if len(dones) > 0 else False
                print(f"  Def {i} to Att: dist={dist:.4f}, sensing_radius={scenario.sensing_radius}, sensed={sensed}, done={done}")
                
                # Check sensing rewards
                if hasattr(scenario, 'attacker_sensing_rewards'):
                    sensing_reward = scenario.attacker_sensing_rewards[0, 0].item()
                    print(f"    Sensing reward: {sensing_reward}")
        
        # Force episode to end after sensing for testing
        if hasattr(scenario, 'attacker_sensed') and scenario.attacker_sensed[0, 0].item():
            print(f"Step {step}: FORCING EPISODE END - ATTACKER SENSED!")
            break
        
        # Print positions every 5 steps
        if step % 5 == 0:
            print(f"\nStep {step} positions:")
            for i, agent in enumerate(env.agents):
                vmas_pos = agent.state.pos[0].numpy()
                global_pos = scenario._vmas_to_world(vmas_pos)
                agent_type = "Def" if agent.is_defender else "Att"
                print(f"  {agent_type} {agent.name}: Global {global_pos}")
        
        if dones[0]:
            print(f"Episode done at step {step}")
            break
        
        total_steps += 1
    
    print(f"\n=== Summary ===")
    print(f"Total steps simulated: {total_steps}")
    print(f"Sensing events: {sensing_events}")
    print(f"Final rewards: {[r[0].item() for r in rewards]}")
    
    # Check distances
    print(f"\n=== Final Distances ===")
    defenders = [a for a in env.agents if a.is_defender]
    attackers = [a for a in env.agents if not a.is_defender]
    
    for att in attackers:
        att_pos = att.state.pos[0]
        for def_agent in defenders:
            def_pos = def_agent.state.pos[0]
            dist = torch.norm(att_pos - def_pos).item()
            print(f"Distance {att.name} to {def_agent.name}: {dist:.4f} (sensing radius: {scenario.sensing_radius})")

if __name__ == "__main__":
    debug_environment()