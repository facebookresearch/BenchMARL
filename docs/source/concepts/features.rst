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
This allows you to change some parts of the config (e.g., task parameters to evaluate in a new setting).

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

Evaluation is automatically run throughout training and can be configured from :class:`~benchmarl.experiment.ExperimentConfig`.
By default, evaluation will be run in different domain randomised environments throughout training.
If you want to always evaluate in the same exact (seeded) environments, set :attr:`benchmarl.experiment.ExperimentConfig.evaluation_static`.

To evaluate a saved experiment, you can:

.. code-block:: python

   from benchmarl.hydra_config import reload_experiment_from_file
   experiment = reload_experiment_from_file(checkpoint_file)
   experiment.evaluate()

This will run an iteration of evaluation, logging it to the experiment loggers (and to json if :attr:`benchmarl.experiment.ExperimentConfig.create_json` ``=True``).

There is a command line script which automates this:

.. code-block:: console

   python benchmarl/evaluate.py ../outputs/2024-09-09/20-39-31/mappo_balance_mlp__cd977b69_24_09_09-20_39_31/checkpoints/checkpoint_100.pt

Rendering
---------

Rendering is performed by default during evaluation (:py:attr:`benchmarl.experiment.ExperimentConfig.render` ``= True``).
If multiple evaluation episodes are requested (:py:attr:`benchmarl.experiment.ExperimentConfig.evaluation_episodes` ``>1``), then only the first one will be rendered.

Renderings will be made available in the loggers you chose (:py:attr:`benchmarl.experiment.ExperimentConfig.loggers`):

- In Wandb, renderings are reported under ``eval/video``
- In CSV, renderings are saved in the experiment folder under ``video``

Devices
-------

It is possible to choose different devices for simulation, training, and buffer storage (in the off-policy case).

These devices can be any :class:`torch.device` and are set via :attr:`benchmarl.experiment.ExperimentConfig.sampling_device`,
:attr:`benchmarl.experiment.ExperimentConfig.train_device`, :attr:`benchmarl.experiment.ExperimentConfig.buffer_device`.

:attr:`~benchmarl.experiment.ExperimentConfig.buffer_device` can also be set to ``"disk"`` to store buffers on disk.

Note that for vectorized simulators such as `VMAS <https://github.com/proroklab/VectorizedMultiAgentSimulator>`__, choosing
:attr:`~benchmarl.experiment.ExperimentConfig.sampling_device` ``="cuda"`` and :attr:`~benchmarl.experiment.ExperimentConfig.train_device` ``="cuda"``
will give important speed-ups as both simulation and training will be run in a batch on the GPU, with no data being moved around.

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
