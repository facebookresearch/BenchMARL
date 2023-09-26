# BenchMARL
[![tests](https://github.com/facebookresearch/BenchMARL/actions/workflows/unit_tests.yml/badge.svg)](test)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/downloads/)

```bash
python benchmarl/run.py algorithm=mappo task=vmas/balance
```

BenchMARL is a Multi-Agent Reinforcement Learning (MARL) training library created to enable reproducibility
and benchmarking across different MARL algorithms and environments.
Its mission is to present a standardized interface that allows easy integration of new algorithms and environments to 
provide a fair comparison with existing solutions.
BenchMARL uses [TorchRL](https://github.com/pytorch/rl) as its backend, which grants it high performance 
and state-of-the-art implementations. 
BenchMARL data reporting is compatible with [marl-eval](https://github.com/instadeepai/marl-eval) for standardized and
statistically strong evaluations.

## Table of contents

- [BenchMARL](#benchmarl)
  * [Table of contents](#table-of-contents)
  * [Concept](#concept)
    + [Experiment](#experiment)
    + [Algorithms](#algorithms)
    + [Environments](#environments)
    + [Models](#models)
  * [How to use](#how-to-use)
    + [Notebooks](#notebooks)
    + [Install](#install)
    + [Run](#run)
  * [Reporting and plotting](#reporting-and-plotting)
  * [Extending](#extending)
  * [Configuring](#configuring)
    + [Algorithm](#algorithm)
    + [Task](#task)
    + [Model](#model)
      - [Sequence model](#sequence-model)
  * [Features](#features)
    + [Logging](#logging)
    + [Checkpointing](#checkpointing)
    + [Callbacks](#callbacks)


## Concept

### Experiment
### Algorithms
### Environments
### Models

## How to use
### Notebooks
### Install
### Run

## Reporting and plotting

## Extending


## Configuring

Running custom experiments is extremely simplified by the [Hydra](https://hydra.cc/) configurations.

The default configuration for the library is contained in the [`conf`](benchmarl/conf) folder.

To run an experiment, you need to select a task and an algorithm
```bash
python hydra_run.py task=vmas/balance algorithm=mappo
```
You can run a set of experiments. For example like this
```bash
python hydra_run.py --multirun task=vmas/balance algorithm=mappo,maddpg,masac,qmix
```

### Algorithm

You will need to specify an algorithm when launching your hydra script.

```bash
python hydra_run.py algorithm=mappo
```

Available ones and their configs can be found at [`conf/algorithm`](benchmarl/conf/algorithm).

We suggest to not modify the algorithms config when running your benchmarks in order to guarantee
reproducibility. 

### Task

You will need to specify a task when launching your hydra script.

Available ones and their configs can be found at [`conf/task`](benchmarl/conf/task) and are sorted
in enviornment folders.

We suggest to not modify the tasks config when running your benchmarks in order to guarantee
reproducibility. 

```bash
python hydra_run.py task=vmas/balance
```

### Model

By default an MLP model will be loaded with the default config.

Available models and their configs can be found at [`conf/model/layers`](benchmarl/conf/model/layers).

```bash
python hydra_run.py model=layers/mlp
```


#### Sequence model
To use the sequence model. Available layer names are in the [`conf/model/layers`](benchmarl/conf/model/layers) folder.
```bash
python hydra_run.py "model=sequence" "model.intermediate_sizes=[256]" "model/layers@model.layers.l1=mlp" "model/layers@model.layers.l2=mlp" 
```
Adding a layer
```bash
python hydra_run.py "+model/layers@model.layers.l3=mlp"
```
Removing a layer
```bash
python hydra_run.py "~model.layers.l2"
```
Configuring a layer
```bash
python hydra_run.py "model.layers.l1.num_cells=[3]"
```


## Features

### Logging

BenchMARL is compatible with the [TorchRL loggers](https://github.com/pytorch/rl/tree/main/torchrl/record/loggers).
A list of logger names can be provided in the [experiment config](benchmarl/conf/experiment/base_experiment.yaml).
Example of available options are: `wandb`, `csv`, `mflow`, `tensorboard` or any other option available in TorchRL. You can specify the loggers
in the yaml config files or in the script arguments like so:
```bash
python hydra_run.py "experiment.loggers=[wandb]"
```

Additionally, you can specify a `create_json` argument which instructs the trainer to output a `.json` file in the
format specified by [marl-eval](https://github.com/instadeepai/marl-eval).

### Checkpointing 
### Callbacks
