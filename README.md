# BenchMARL




## Hydra config

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
