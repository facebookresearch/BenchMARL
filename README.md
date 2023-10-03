# BenchMARL
[![tests](https://github.com/facebookresearch/BenchMARL/actions/workflows/unit_tests.yml/badge.svg)](test)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/downloads/)

```bash
python benchmarl/run.py algorithm=mappo task=vmas/balance
```

[![Examples](https://img.shields.io/badge/Examples-blue.svg)](examples) 
<!--
[![Static Badge](https://img.shields.io/badge/Benchmarks-Wandb-yellow)]()
-->


BenchMARL is a Multi-Agent Reinforcement Learning (MARL) training library created to enable reproducibility
and benchmarking across different MARL algorithms and environments.
Its mission is to present a standardized interface that allows easy integration of new algorithms and environments to 
provide a fair comparison with existing solutions.
BenchMARL uses [TorchRL](https://github.com/pytorch/rl) as its backend, which grants it high performance 
and state-of-the-art implementations. 
BenchMARL data reporting is compatible with [marl-eval](https://github.com/instadeepai/marl-eval) for standardized and
statistically strong evaluations.

- [BenchMARL](#benchmarl)
  * [How to use](#how-to-use)
    + [Notebooks](#notebooks)
    + [Install](#install)
    + [Run](#run)
  * [Concept](#concept)
  * [Reporting and plotting](#reporting-and-plotting)
  * [Extending](#extending)
  * [Configuring](#configuring)
    + [Algorithm](#algorithm)
    + [Task](#task)
    + [Model](#model)
  * [Features](#features)
    + [Logging](#logging)
    + [Checkpointing](#checkpointing)
    + [Callbacks](#callbacks)


## How to use

### Notebooks

### Install

#### Install TorchRL

You can install TorchRL from PyPi.

```bash
pip install torchrl
```
For more details, or for installing nightly versions, see the
[TorchRL installation guide](https://github.com/pytorch/rl#installation).

#### Install BenchMARL
You can just install it from github
```bash
pip install git+https://github.com/facebookresearch/BenchMARL
```
Or also clone it locally to access the configs and scripts
```bash
git clone https://github.com/facebookresearch/BenchMARL.git
pip install -e BenchMARL
```
#### Install environments

All enviornment dependencies are optional in BenchMARL and can be installed separately.

##### VMAS

```bash
pip install vmas
```

##### PettingZoo
```bash
pip install "pettingzoo[all]"
```

##### SMACv2

Follow the instructions on the environment [repository](https://github.com/oxwhirl/smacv2).

[Here](.github/unittest/install_smacv2.sh) is how we install it on linux.

### Run

Experiments are launched with a [default configuration](benchmarl/conf) that 
can be overridden in many ways. 
To learn how to customize and override configurations
please refer to the [configuring section](#configuring).

#### Command line

To launch an experiment from the command line you can do

```bash
python benchmarl/run.py algorithm=mappo task=vmas/balance
```
[![Example](https://img.shields.io/badge/Example-blue.svg)](examples/running/run_experiment.sh)


Thanks to [hydra](https://hydra.cc/docs/intro/), you can run benchmarks as multi-runs like:
```bash
python benchmarl/run.py -m algorithm=mappo,qmix,masac task=vmas/balance,vmas/sampling seed=0,1
```
[![Example](https://img.shields.io/badge/Example-blue.svg)](examples/running/run_benchmark.sh)

The default implementation for hydra multi-runs is sequential, but [parallel execution is
also available](https://hydra.cc/docs/plugins/joblib_launcher/).

#### Script

You can also load and launch your experiments from within a script

```python
 experiment = Experiment(
    task=VmasTask.BALANCE.get_from_yaml(),
    algorithm_config=MappoConfig.get_from_yaml(),
    model_config=MlpConfig.get_from_yaml(),
    critic_model_config=MlpConfig.get_from_yaml(),
    seed=0,
    config=ExperimentConfig.get_from_yaml(),
)
experiment.run()
```
[![Example](https://img.shields.io/badge/Example-blue.svg)](examples/running/run_experiment.py)


You can also run multiple experiments in a `Benchmark`.

```python
benchmark = Benchmark(
    algorithm_configs=[
        MappoConfig.get_from_yaml(),
        QmixConfig.get_from_yaml(),
        MasacConfig.get_from_yaml(),
    ],
    tasks=[
        VmasTask.BALANCE.get_from_yaml(),
        VmasTask.SAMPLING.get_from_yaml(),
    ],
    seeds={0, 1},
    experiment_config=ExperimentConfig.get_from_yaml(),
    model_config=MlpConfig.get_from_yaml(),
    critic_model_config=MlpConfig.get_from_yaml(),
)
benchmark.run_sequential()
```
[![Example](https://img.shields.io/badge/Example-blue.svg)](examples/running/run_benchmark.py)


## Concept

The goal of BenchMARL is to bring different MARL environments and algorithms
under the same interfaces to enable fair and reproducible comaprison and benchmarking.
BenchMARL is a full-pipline unified training library with the goal of enabling users to run
any comparison they want across our algorithms and tasks in just one line of code.
To achieve this, BenchMARL interconnects components from [TorchRL](https://github.com/pytorch/rl), 
which provides an efficient and reliable backend.

The library has a [default configuration](benchmarl/conf) for each of its components.
While parts of this configuration are supposed to be changed (for example experiment configurations),
other parts (such as tasks) should not be changed to allow for reproducibility.
To aid in this, each version of BenchMARL is paired to a default configuration.

Let's now introduce each component in the library.

**Experiment**. An experiment is a training run in which an algorithm, a task, and a model are fixed.
Experiments are configured by passing these values alongside a seed and the experiment hyperparameters.
The experiment [hyperparameters](benchmarl/conf/experiment/base_experiment.yaml) cover both 
on-policy and off-policy algorithms, discrete and continuous actions, and probabilistic and deterministic policies
(as they are agnostic of the algorithm or task used).
An experiment can be launched from the command line or from a script. 
See the [run](#run) section for more information.

**Benchmark**. In the library we call `benchmark` a collection of experiments that can vary in tasks, algorithm, or model.
A benchmark shares the same experiment configuration across all of its experiments.
Benchmarks allow to compare different MARL components in a standardized way.
A benchmark can be launched from the command line or from a script. 
See the [run](#run) section for more information.

**Algorithms**. Algorithms are an ensemble of components (e.g., losss, replay buffer) which
determine the training strategy. Here is a table with the currently implemented algorithms in BenchMARL.

| Name                                   | On/Off policy | Actor-critic | Full-observability in critic | Action compatibility          | Probabilistic actor |   
|----------------------------------------|---------------|--------------|------------------------------|-------------------------------|---------------------|
| [MAPPO](https://arxiv.org/abs/2103.01955)                              | On            | Yes          | Yes                          | Continuous + Discrete         | Yes                 |   
| [IPPO](https://arxiv.org/abs/2011.09533)                               | On            | Yes          | No                           | Continuous + Discrete         | Yes                 |  
| [MADDPG](https://arxiv.org/abs/1706.02275)                             | Off           | Yes          | Yes                          | Continuous                    | No                  | 
| [IDDPG](benchmarl/algorithms/iddpg.py) | Off           | Yes          | No                           | Continuous                    |  No                 |   
| [MASAC](benchmarl/algorithms/masac.py) | Off           | Yes          | Yes                          | Continuous + Discrete         |  Yes                |   
| [ISAC](benchmarl/algorithms/isac.py)   | Off           | Yes          | No                           | Continuous + Discrete         |  Yes                |   
| [QMIX](https://arxiv.org/abs/1803.11485)                               | Off           | No           | NA                           | Discrete                      |  No                 | 
| [VDN](https://arxiv.org/abs/1706.05296)                                | Off           | No           | NA                           | Discrete                      |  No                 |  
| [IQL](https://www.semanticscholar.org/paper/Multi-Agent-Reinforcement-Learning%3A-Independent-Tan/59de874c1e547399b695337bcff23070664fa66e)                                | Off           | No           | NA                           | Discrete                      |  No                 |  


**Tasks**. Tasks are scenarios from a specific environment which constitute the MARL
challange to solve. They differe based on many aspects, here is a table with the current environments in BenchMARL

| Enviromnent | Tasks                                 | Cooperation               | Global state | Reward function               | 
|-------------|---------------------------------------|---------------------------|--------------|-------------------------------|
| [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) | [TBC](benchmarl/conf/task/vmas)       | Cooperative + Competitive | No           | Shared + Independent + Global |  
| [SMAC](https://github.com/oxwhirl/smac)   | [TBC](benchmarl/conf/task/smacv2)     | Cooperative               | Yes          | Global                        |  
| [SMACv2](https://github.com/oxwhirl/smacv2) | [TBC](benchmarl/conf/task/smacv2)     | Cooperative               | Yes          | Global                        |  
| [MPE](https://github.com/openai/multiagent-particle-envs)     | [TBC](benchmarl/conf/task/pettingzoo) | Cooperative + Competitive | Yes          | Shared + Independent          |   
| [SISL](https://github.com/sisl/MADRL)    | [TBC](benchmarl/conf/task/pettingzoo)       | Cooperative               | No           | Shared                        |  

**Models**. Models are neural networks used to process data. They can be used as actors (policies) or, 
when possible, as critics. We provide a set of base models (layers) and a SequenceModel to concatenate
different. All the models can be used with or without parameter sharing within an 
agent group. Here is a table of the models implemented in BenchMARL

| Name | Decentralized | Centralized with local inputs | Centralized with global input | 
|------|:-------------:|:-----------------------------:|:-----------------------------:|
| MLP  |       ✅       |               ✅               |               ✅               | 

And the ones that are _work in progress_

| Name | Decentralized | Centralized with local inputs | Centralized with global input | 
|------|:-------------:|:-----------------------------:|:-----------------------------:|
| GNN  |       ✅       |               ✅               |               ❌               | 
| CNN  |       ✅       |               ✅               |               ✅               | 


## Reporting and plotting
TBC

## Extending
TBC


## Configuring
Experiment configurations are in [`benchmarl/conf/config.yaml`](benchmarl/conf/config.yaml),
with the experiment hyperparameters in [`benchmarl/conf/experiment`](benchmarl/conf/experiment).


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
TBC
### Callbacks
TBC
