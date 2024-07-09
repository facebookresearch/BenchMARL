# Using Weights & Biases (W&B) Sweeps with BenchMARL

You can improve the performance of your RL agents with hyperparameter tuning. It's easy to train multiple models with different hyperparameters using hyperparameter sweep on W&B with BenchMARL and Hydra. Modify `sweepconfig.yaml` to define your sweep configuration and run it from the command line.

## Prerequisites

- Ensure you have Weights & Biases: `pip install wandb` installed on top of benchmarl requirements.

- Update the `benchmarl/conf/config.yaml` with your desired experiment setup, e.g.:

```yaml
defaults:
  - experiment: base_experiment
  - algorithm: ippo
  - task: customenv/task_1
  - model: layers/mlp
  - model@critic_model: layers/mlp
  - _self_

seed: 0
```

## Step 1: Define Your Sweep Configuration

First, create or modify the `sweepconfig.yaml` file. Check the [W&B Sweep Configuration Documentation](https://docs.wandb.ai/guides/sweeps/sweep-config-keys) for detailed configuration options.


The YAML file already contains the basic elements required to work with BenchMARL. Change the values according to your desired experiment setup. Note that the parameters in the YAML file should use dots (e.g., `experiment.lr`) rather than standard double nested configurations ([like in this community discussion](https://community.wandb.ai/t/nested-sweep-configuration/3369)) since you are using Hydra.


```yaml
entity: "ENTITY_NAME"

#options: bayes, random, grid
method: bayes

metric:
  name: eval/agent/reward/episode_reward_mean
  goal: maximize

parameters:
  experiment.lr:
    max: 0.003
    min: 0.000025
    # distribution: uniform

  experiment.max_n_iters:
    value: 321

```

## Step 2: Initialize sweep

To run the sweep, initialize it using the following command in your terminal:

```bash
wandb sweep sweepconfig.yaml
```

W&B will automatically create a sweep and return a command for you to run, like:

```bash
wandb: Created sweep with ID: xyz123
wandb: View sweep at: https://wandb.ai/your_entity/your_project/sweeps/xyz123
wandb: Run sweep agent with: wandb agent your_entity/your_project/xyz123
```

## Step 3: Start sweep agents
Run the command provided in the terminal to start the sweep agent:

```bash
wandb agent mc-team/project-name/xyz123
```

This will start the agent and begin running experiments according to your sweep configuration.

## References

https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw?galleryTag=posts&utm_source=fully_connected&utm_medium=blog&utm_campaign=hydra

https://docs.wandb.ai/guides/sweeps
