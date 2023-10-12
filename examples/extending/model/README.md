
# Creating a new model

Here are the steps to create a new model. 

1. Create your `CustomModel` and `CustomModelConfig` following the example
in [`custom_model.py`](custom_model.py). These will be the model code
and an associated dataclass to validate loaded configs.
2. Create a `custommodel.yaml` with the configuration parameters you defined 
in your script. Make sure it has a `name` entry equal to `custom_model` to let hydra know which python dataclass it is 
associated to. You can see [`custommodel.yaml`](custommodel.yaml)
for an example.
3. Place your model script in [`benchmarl/models`](../../../benchmarl/models) and 
your config in [`benchmarl/conf/model/layers`](../../../benchmarl/conf/model/layers) (or any other place you want to 
override from)
4. Add `{"custom_model": CustomModelConfig}` to the [`benchmarl.models.model_config_registry`](../../../benchmarl/models/__init__.py)
5. Load it with
```bash
python benchmarl/run.py model=layers/custommodel algorithm=... task=...
```
