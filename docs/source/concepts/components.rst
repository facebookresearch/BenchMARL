Components
==========

The goal of BenchMARL is to bring different MARL environments and algorithms
under the same interfaces to enable fair and reproducible comparison and benchmarking.
BenchMARL is a full-pipline unified training library with the goal of enabling users to run
any comparison they want across our algorithms and tasks in just one line of code.
To achieve this, BenchMARL interconnects components from :torchrl:`null` `TorchRL <https://github.com/pytorch/rl>`__,
which provides an efficient and reliable backend.

The library has a `default configuration <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf>`__ for each of its components.
While parts of this configuration are supposed to be changed (for example experiment configurations),
other parts (such as tasks) should not be changed to allow for reproducibility.
To aid in this, each version of BenchMARL is paired to a default configuration.

Let's now introduce each component in the library.

Experiment
----------

An :class:`~benchmarl.experiment.Experiment` is a training run in which an  :class:`~benchmarl.algorithms.Algorithm`, :class:`~benchmarl.environments.Task`,
and a :class:`~benchmarl.models.Model` are fixed.
Experiments are configured by passing these values alongside a seed and the experiment hyperparameters.
The experiment `hyperparameters <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/experiment/base_experiment.yaml>`__ cover both
on-policy and off-policy algorithms, discrete and continuous actions, and probabilistic and deterministic policies
(as they are agnostic of the algorithm or task used).
An experiment can be launched from the command line or from a script.
See the :doc:`/usage/running` section for more information.

Benchmark
---------

In the library we call benchmark a collection of experiments that can vary in tasks, algorithm, or model.
A benchmark shares the same experiment configuration across all of its experiments.
Benchmarks allow to compare different MARL components in a standardized way.
A benchmark can be launched from the command line or from a script.
See the [run](#run) section for more information.

Algorithms
------------

Algorithms are an ensemble of components (e.g., loss, replay buffer) which
determine the training strategy. Here is a table with the currently implemented algorithms in BenchMARL.

.. _algorithm-table:

.. table:: Algorithms in BenchMARL

    +---------------------------------------+---------------+--------------+------------------------------+-----------------------+---------------------+--+
    |                       Algorithm       | On/Off policy | Actor-critic | Full-observability in critic | Action compatibility  | Probabilistic actor |  |
    +=======================================+===============+==============+==============================+=======================+=====================+==+
    | :class:`~benchmarl.algorithms.Mappo`  |      On       |     Yes      |             Yes              | Continuous + Discrete |         Yes         |  |
    +---------------------------------------+---------------+--------------+------------------------------+-----------------------+---------------------+--+
    | :class:`~benchmarl.algorithms.Ippo`   |      On       |     Yes      |              No              | Continuous + Discrete |         Yes         |  |
    +---------------------------------------+---------------+--------------+------------------------------+-----------------------+---------------------+--+
    | :class:`~benchmarl.algorithms.Maddpg` |      Off      |     Yes      |             Yes              |      Continuous       |         No          |  |
    +---------------------------------------+---------------+--------------+------------------------------+-----------------------+---------------------+--+
    | :class:`~benchmarl.algorithms.Iddpg`  |      Off      |     Yes      |              No              |      Continuous       |         No          |  |
    +---------------------------------------+---------------+--------------+------------------------------+-----------------------+---------------------+--+
    | :class:`~benchmarl.algorithms.Masac`  |      Off      |     Yes      |             Yes              | Continuous + Discrete |         Yes         |  |
    +---------------------------------------+---------------+--------------+------------------------------+-----------------------+---------------------+--+
    | :class:`~benchmarl.algorithms.Isac`   |      Off      |     Yes      |              No              | Continuous + Discrete |         Yes         |  |
    +---------------------------------------+---------------+--------------+------------------------------+-----------------------+---------------------+--+
    | :class:`~benchmarl.algorithms.Qmix`   |      Off      |      No      |              NA              |       Discrete        |         No          |  |
    +---------------------------------------+---------------+--------------+------------------------------+-----------------------+---------------------+--+
    | :class:`~benchmarl.algorithms.Vdn`    |     Off       |      No      |              NA              |       Discrete        |         No          |  |
    +---------------------------------------+---------------+--------------+------------------------------+-----------------------+---------------------+--+
    | :class:`~benchmarl.algorithms.Iql`    |     Off       |      No      |              NA              |       Discrete        |         No          |  |
    +---------------------------------------+---------------+--------------+------------------------------+-----------------------+---------------------+--+

Environments
------------

Tasks are scenarios from a specific environment which constitute the MARL
challenge to solve.
They differ based on many aspects, here is a table with the current environments in BenchMARL:


.. _environment-table:

.. table:: Environments in BenchMARL

    +-------------------------------------------------+-------+---------------------------+--------------+-------------------------------+-----------------------+------------+
    | Environment                                     | Tasks |        Cooperation        | Global state |        Reward function        |     Action space      | Vectorized |
    +=================================================+=======+===========================+==============+===============================+=======================+============+
    |    :class:`~benchmarl.environments.VmasTask`    |  27   | Cooperative + Competitive |      No      | Shared + Independent + Global | Continuous + Discrete |    Yes     |
    +-------------------------------------------------+-------+---------------------------+--------------+-------------------------------+-----------------------+------------+
    |   :class:`~benchmarl.environments.Smacv2Task`   |  15   |        Cooperative        |     Yes      |            Global             |       Discrete        |     No     |
    +-------------------------------------------------+-------+---------------------------+--------------+-------------------------------+-----------------------+------------+
    | :class:`~benchmarl.environments.PettingZooTask` |  10   | Cooperative + Competitive |   Yes + No   |     Shared + Independent      | Continuous + Discrete |     No     |
    +-------------------------------------------------+-------+---------------------------+--------------+-------------------------------+-----------------------+------------+
    | :class:`~benchmarl.environments.MeltingPotTask` |  49   | Cooperative + Competitive |     Yes      |          Independent          |       Discrete        |     No     |
    +-------------------------------------------------+-------+---------------------------+--------------+-------------------------------+-----------------------+------------+
    | :class:`~benchmarl.environments.MAgentTask`     |  1    | Cooperative + Competitive |     Yes      |          Global in groups     |       Discrete        |     No     |
    +-------------------------------------------------+-------+---------------------------+--------------+-------------------------------+-----------------------+------------+



Models
------

Models are neural networks used to process data. They can be used as actors (policies) or,
when requested, as critics. We provide a set of base models (layers) and a :class:`~benchmarl.models.SequenceModel` to concatenate
different layers. All the models can be used with or without parameter sharing within an
agent group. Here is a table of the models implemented in BenchMARL


.. _model-table:

.. table:: Models in BenchMARL

    +-------------------------------------+---------------+-------------------------------+-------------------------------+
    | Name                                | Decentralized | Centralized with local inputs | Centralized with global input |
    +=====================================+===============+===============================+===============================+
    | :class:`~benchmarl.models.Mlp`      |      Yes      |              Yes              |              Yes              |
    +-------------------------------------+---------------+-------------------------------+-------------------------------+
    | :class:`~benchmarl.models.Gru`      |      Yes      |              Yes              |              Yes              |
    +-------------------------------------+---------------+-------------------------------+-------------------------------+
    | :class:`~benchmarl.models.Lstm`     |      Yes      |              Yes              |              Yes              |
    +-------------------------------------+---------------+-------------------------------+-------------------------------+
    | :class:`~benchmarl.models.Gnn`      |      Yes      |              Yes              |              No               |
    +-------------------------------------+---------------+-------------------------------+-------------------------------+
    | :class:`~benchmarl.models.Cnn`      |      Yes      |              Yes              |              Yes              |
    +-------------------------------------+---------------+-------------------------------+-------------------------------+
    | :class:`~benchmarl.models.Deepsets` |      Yes      |              Yes              |              Yes              |
    +-------------------------------------+---------------+-------------------------------+-------------------------------+
