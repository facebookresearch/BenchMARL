# BenchMARL Target Defense Implementation

## Overview
This repository contains a complete implementation of a multi-agent target defense environment for the BenchMARL framework. Defenders must coordinate to intercept attackers before they reach a protected target line, using game-theoretic optimal rewards from Apollonius circle optimization.

## Features

### 🎯 Environment
- **Multi-agent coordination**: 1v1, 2v1, 3v1 scenarios
- **Game-theoretic rewards**: Apollonius solver integration for optimal feedback
- **Configurable parameters**: Sensing radius, speed ratios, spawn configurations
- **Enhanced observations**: Relative positioning for coordination learning
- **Area-based spawning**: Realistic attacker spawn zones (NEW!)

### 🎬 Visualization
- **Enhanced wandb videos**: Sensing circles, spawn markers, trajectory trails
- **Real trajectory analysis**: Actual learned agent movement patterns
- **Performance plots**: Learning curves, sensing rates, parameter comparisons
- **Animated GIFs**: Strategy visualization and exploration analysis

### 🤖 Algorithm Support
- **MAPPO**: Multi-Agent Proximal Policy Optimization
- **MADDPG**: Multi-Agent Deep Deterministic Policy Gradient
- **MASAC**: Multi-Agent Soft Actor-Critic

## Quick Start

### Installation
```bash
pip install torch vmas tensordict torchrl hydra-core wandb cvxpy
git clone https://github.com/YOUR_USERNAME/BenchMARL-Target-Defense.git
cd BenchMARL-Target-Defense
pip install -e .
```

### Basic Training
```bash
# Standard 3v1 training
python -m benchmarl.run task=vmas/target_defense algorithm=mappo \\
  experiment.max_n_frames=1200000 \\
  experiment.render=true

# Area spawning mode
python -m benchmarl.run task=vmas/target_defense algorithm=mappo \\
  task.spawn_area_mode=true \\
  task.spawn_area_width=0.2 \\
  experiment.render=true
```

### Parameter Sweeps
```bash
# Algorithm and speed comparison
python -m benchmarl.run -m \\
  task=vmas/target_defense \\
  algorithm=mappo,maddpg,masac \\
  task.speed_ratio=0.1,0.2,0.3,0.4 \\
  experiment.max_n_frames=1800000
```

## Configuration

### Key Parameters
- **num_defenders**: Number of defender agents (1-5)
- **num_attackers**: Number of attacker agents (1-3)  
- **sensing_radius**: Detection range (0.05-0.3)
- **speed_ratio**: Attacker speed / defender speed (0.1-0.8)
- **spawn_area_mode**: Area vs discrete spawning
- **spawn_area_width**: Spawn zone height (0.1-0.4)

### Example Configurations
```yaml
# Challenging 1v1 with area spawning
num_defenders: 1
num_attackers: 1
sensing_radius: 0.1
speed_ratio: 0.3
spawn_area_mode: true
spawn_area_width: 0.2

# Standard 3v1 with discrete spawns  
num_defenders: 3
num_attackers: 1
sensing_radius: 0.15
speed_ratio: 0.2
spawn_area_mode: false
num_spawn_positions: 6
```

## Results

### Performance Achieved
- **Best sensing success**: 1.47% with area spawning
- **Highest rewards**: 0.359 mean defender reward
- **Successful coordination**: Multi-agent spatial coverage
- **Algorithm comparison**: MAPPO > MADDPG > MASAC

### Research Insights
- **Area spawning increases difficulty**: More realistic tactical scenarios
- **Enhanced observations crucial**: Prevents defender clustering
- **Parameter sensitivity**: Small sensing radius changes have large impact
- **Game-theoretic integration**: Successful Apollonius reward system

## File Structure
```
├── benchmarl/environments/vmas/target_defense.py    # Main environment
├── benchmarl/conf/task/vmas/target_defense.yaml     # Configuration  
├── apollonius_solver.py                            # Game theory solver
├── area_spawning_tests/                             # Validation tests
├── BenchMARL_Target_Defense_Report.tex             # Research documentation
└── HOPPER_INSTALLATION_GUIDE.md                    # Server deployment
```

## Citation
If you use this environment in your research, please cite:

```bibtex
@misc{das2025benchmarl_target_defense,
  title={Multi-Agent Target Defense Environment for BenchMARL},
  author={Goutam Das},
  year={2025},
  institution={George Mason University}
}
```

## License
This implementation extends the BenchMARL framework and follows the same licensing terms.