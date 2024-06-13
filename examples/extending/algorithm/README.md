
# Creating a new algorithm

Here are the steps to create a new algorithm.

1. Create your `CustomAlgorithm` and `CustomAlgorithmConfig` following the example
in [`algorithms/customalgorithm.py`](algorithms/customalgorithm.py). These will be the algorithm code
and an associated dataclass to validate loaded configs.
2. Create a `customalgorithm.yaml` with the configuration parameters you defined 
in your script. Make sure it has `customalgorithm_config` within its defaults at 
the top of the file to let hydra know which python dataclass it is 
associated to. You can see [`conf/algorithm/customalgorithm.yaml`](conf/algorithm/customalgorithm.yaml)
for an example.
3. Place your algorithm script in [`benchmarl/algorithms`](../../../benchmarl/algorithms) and 
your config in [`benchmarl/conf/algorithm`](../../../benchmarl/conf/algorithm) (or any other place you want to 
override from)
4. Add `{"customalgorithm": CustomAlgorithmConfig}` to the [`benchmarl.algorithms.algorithm_config_registry`](../../../benchmarl/algorithms/__init__.py)
5. Load it with
```bash
python benchmarl/run.py algorithm=customalgorithm task=...
```
