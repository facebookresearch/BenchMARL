Features
========

BenchMARL has several features:

- A test CI with integration and training test routines that are run for all simulators and algorithms
- Integration in the official TorchRL ecosystem for dedicated support


Logging
-------

BenchMARL is compatible with the `TorchRL loggers <https://github.com/pytorch/rl/tree/main/torchrl/record/loggers>`__.
A list of logger names can be provided in the `experiment config <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/experiment/base_experiment.yaml>`__.
Example of available options are: ``wandb``, ``csv``, ``mflow``, ``tensorboard`` or any other option available in :torchrl:`null` `TorchRL <https://github.com/pytorch/rl>`__.
You can specify the loggers
in the yaml config files or in the script arguments like so:

.. code-block:: console

    python benchmarl/run.py algorithm=mappo task=vmas/balance "experiment.loggers=[wandb]"

The :wandb:`null` `wandb <https://wandb.ai/>`__ logger is fully compatible with experiment restoring and will automatically resume the run of
the loaded experiment.

Checkpointing
-------------

Experiments can be checkpointed every ``experiment.checkpoint_interval`` collected frames.
Experiments will use an output folder for logging and checkpointing which can be specified in ``experiment.save_folder``.
If this is left unspecified,
the default will be the hydra output folder (if using hydra) or (otherwise) the current directory
where the script is launched.
The output folder will contain a folder for each experiment with the corresponding experiment name.
Their checkpoints will be stored in a ``"checkpoints"`` folder within the experiment folder.

.. code-block:: console

   python benchmarl/run.py task=vmas/balance algorithm=mappo experiment.max_n_iters=3 experiment.on_policy_collected_frames_per_batch=100 experiment.checkpoint_interval=100

.. python_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/checkpointing/reload_experiment.py

Reloading
---------

To load from a checkpoint, you can do it in multiple ways:

You can pass the absolute checkpoint file name to ``experiment.restore_file``.

.. code-block:: console

   python benchmarl/run.py task=vmas/balance algorithm=mappo experiment.max_n_iters=6 experiment.on_policy_collected_frames_per_batch=100 experiment.restore_file="/hydra/experiment/folder/checkpoint/checkpoint_300.pt"

.. python_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/checkpointing/reload_experiment.py

If you do not need to change the config, you can also just resume from the checkpoint file with:

.. code-block:: console

   python benchmarl/resume.py ../outputs/2024-09-09/20-39-31/mappo_balance_mlp__cd977b69_24_09_09-20_39_31/checkpoints/checkpoint_100.pt

In Python, this is equivalent to:

.. code-block:: python

   from benchmarl.hydra_config import reload_experiment_from_file
   experiment = reload_experiment_from_file(checkpoint_file)
   experiment.run()


Evaluating
----------

To evaluate an experiment, you can:

.. code-block:: python

   from benchmarl.hydra_config import reload_experiment_from_file
   experiment = reload_experiment_from_file(checkpoint_file)
   experiment.evaluate()

This will run an iteration of evaluation, logging it to the experiment loggers (and to json if ``create_json==True``.

There is a command line script which automates this:

.. code-block:: console

   python benchmarl/evaluate.py ../outputs/2024-09-09/20-39-31/mappo_balance_mlp__cd977b69_24_09_09-20_39_31/checkpoints/checkpoint_100.pt


Callbacks
---------

Experiments optionally take a list of :class:`~benchmarl.experiment.Callback` which have several methods
that you can implement to see what's going on during training, such
as:

- :py:func:`~benchmarl.experiment.Callback.on_setup`
- :py:func:`~benchmarl.experiment.Callback.on_batch_collected`
- :py:func:`~benchmarl.experiment.Callback.on_train_step`
- :py:func:`~benchmarl.experiment.Callback.on_train_end`
- :py:func:`~benchmarl.experiment.Callback.on_evaluation_end`


.. python_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/callback/custom_callback.py

Ensemble models and algorithms
------------------------------

It is possible to use different algorithms and models for different agent groups.

Ensemble algorithm
^^^^^^^^^^^^^^^^^^

Ensemble algorithms take as input a dictionary mapping group names to algorithm configs:

.. code-block:: python

   from benchmarl.algorithms import EnsembleAlgorithmConfig, IsacConfig, MaddpgConfig

   algorithm_config = EnsembleAlgorithmConfig(
       {"agent": MaddpgConfig.get_from_yaml(), "adversary": IsacConfig.get_from_yaml()}
   )

.. note::
   All algorithms need to be on-policy or off-policy, it is not possible to mix the two paradigms.


.. python_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/ensemble/ensemble_algorithm.py


Ensemble model
^^^^^^^^^^^^^^

Ensemble models take as input a dictionary mapping group names to model configs:

.. code-block:: python

   from benchmarl.models import EnsembleModelConfig, GnnConfig, MlpConfig

   model_config = EnsembleModelConfig(
           {"agent": MlpConfig.get_from_yaml(), "adversary": GnnConfig.get_from_yaml()}
   )


.. note::
   If you use ensemble models with sequence models, make sure the ensemble is the outer layer (you cannot make a sequence of ensembles, but an ensemble of sequences yes).

.. python_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/ensemble/ensemble_model.py
