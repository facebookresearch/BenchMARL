Configuring
===========


As highlighted in the :doc:`/usage/running` section, the project can be configured either
in the script itself or via `hydra <https://hydra.cc/docs/intro/>`__.
We suggest to read the hydra documentation
to get familiar with all its functionalities.

Each component in the project has a corresponding YAML configuration in the BenchMARL
`conf tree <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf>`__.
Components' configurations are loaded from these files into python dataclasses that act
as schemas for validation of parameter names and types. That way we keep the best of
both words: separation of all configuration from code and strong typing for validation!
You can also directly load and validate configuration yaml files without using hydra from a script by calling
``ComponentConfig.get_from_yaml()``.

Experiment
----------

Experiment configurations are in `benchmarl/conf/config.yaml <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/config.yaml>`__.
Running custom experiments is extremely simplified by the `Hydra <https://hydra.cc/docs/intro/>`__ configurations.
The default configuration for the library is contained in the `benchmarl/conf <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf>`__ folder.

When running an experiment you can override its hyperparameters like so

.. code-block:: console

   python benchmarl/run.py task=vmas/balance algorithm=mappo experiment.lr=0.03 experiment.evaluation=true experiment.train_device="cpu"


Experiment hyperparameters are loaded from `benchmarl/conf/experiment <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/experiment>`__
into a dataclass :class:`~benchmarl.experiment.ExperimentConfig` defining their domain.
This makes it so that all and only the parameters expected are loaded with the right types.
You can also directly load them from a script by calling :py:func:`benchmarl.experiment.ExperimentConfig.get_from_yaml`.

Here is an example of overriding experiment hyperparameters from hydra

.. bash_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/configuring/configuring_experiment.sh

or from a script

.. python_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/configuring/configuring_experiment.py

Algorithm
---------

You can override an algorithm configuration when launching BenchMARL.

.. code-block:: console

   python benchmarl/run.py task=vmas/balance algorithm=masac algorithm.num_qvalue_nets=3 algorithm.target_entropy=auto algorithm.share_param_critic=true

Available algorithms and their default configs can be found at `benchmarl/conf/algorithm <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/algorithm>`__.
They are loaded into a dataclass :class:`~benchmarl.algorithm.AlgorithmConfig`, present for each algorithm, defining their domain.
This makes it so that all and only the parameters expected are loaded with the right types.
You can also directly load them from a script by calling ``YourAlgorithmConfig.get_from_yaml()``.

Here is an example of overriding algorithm hyperparameters from hydra

.. bash_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/configuring/configuring_algorithm.sh

or from a script

.. python_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/configuring/configuring_algorithm.py

Task
----

You can override a task configuration when launching BenchMARL.
However this is not recommended for benchmarking as tasks should have fixed version and parameters for reproducibility.

.. code-block:: console

   python benchmarl/run.py task=vmas/balance algorithm=mappo task.n_agents=4


Available tasks and their default configs can be found at `benchmarl/conf/task <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/task>`__.
They are loaded into a dataclass ``TaskConfig``, defining their domain.
Tasks are enumerations under the environment name. For example, :class:`benchmarl.environments.VmasTask.NAVIGATION` represents the navigation task in the
VMAS simulator. This allows autocompletion and seeing all available tasks at once.
You can also directly load them from a script by calling ``YourEnvTask.TASK_NAME.get_from_yaml()``.

Here is an example of overriding task hyperparameters from hydra

.. bash_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/configuring/configuring_task.sh

or from a script

.. python_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/configuring/configuring_task.py

Model
-----

You can override the model configuration when launching BenchMARL.
By default an :class:`~benchmarl.models.Mlp` model will be loaded with the default config.
You can change it like so:

.. code-block:: console

   python benchmarl/run.py task=vmas/balance algorithm=mappo model=layers/mlp model.layer_class="torch.nn.Linear" "model.num_cells=[32,32]" model.activation_class="torch.nn.ReLU"


Available models and their configs can be found at `benchmarl/conf/model/layers <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/model/layers>`__.
They are loaded into a dataclass :class:`~benchmarl.models.ModelConfig`, defining their domain.
You can also directly load them from a script by calling `YourModelConfig.get_from_yaml()`.


Here is an example of overriding model hyperparameters from hydra

.. bash_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/configuring/configuring_model.sh

or from a script

.. python_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/configuring/configuring_model.py


Sequence model
^^^^^^^^^^^^^^

You can compose layers into a sequence model.
Available layer names are in the `benchmarl/conf/model/layers <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/model/layers>`__ folder.

.. code-block:: console

   python benchmarl/run.py task=vmas/balance algorithm=mappo model=sequence "model.intermediate_sizes=[256]" "model/layers@model.layers.l1=mlp" "model/layers@model.layers.l2=mlp" "+model/layers@model.layers.l3=mlp" "model.layers.l3.num_cells=[3]"


Add a layer with ``"+model/layers@model.layers.l3=mlp"``.

Remove a layer with ``"~model.layers.l2"``.

Configure a layer with ``"model.layers.l1.num_cells=[3]"``.


Here is an example of creating a sequence model from hydra

.. bash_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/configuring//configuring_sequence_model.sh

or from a script

.. python_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/configuring/configuring_sequence_model.py
