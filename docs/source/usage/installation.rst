Installation
============


Install TorchRL
---------------

You can install :torchrl:`null` `TorchRL <https://github.com/pytorch/rl>`__ from PyPi.

.. code-block:: console

   pip install torchrl

For more details, or for installing nightly versions, see the
`TorchRL installation guide <https://github.com/pytorch/rl#installation>`__.

Install BenchMARL
-----------------

You can just install it from PyPi

.. code-block:: console

   pip install benchmarl

Or also clone it locally to access the configs and scripts

.. code-block:: console

    git clone https://github.com/facebookresearch/BenchMARL.git
    pip install -e BenchMARL

Install optional packages
-------------------------

By default, BenchMARL has only the core requirements.
Here are some optional packages you may want to install.

Logging
^^^^^^^

You may want to install the following rendering and logging tools

.. code-block:: console

   pip install wandb moviepy

Install environments
--------------------

All environment dependencies are optional in BenchMARL and can be installed separately.

VMAS
^^^^
:github:`null` `GitHub <https://github.com/proroklab/VectorizedMultiAgentSimulator>`__

.. code-block:: console

   pip install vmas

PettingZoo
^^^^^^^^^^
:github:`null` `GitHub <https://github.com/Farama-Foundation/PettingZoo>`__


.. code-block:: console

   pip install "pettingzoo[all]"

MeltingPot
^^^^^^^^^^
:github:`null` `GitHub <https://github.com/google-deepmind/meltingpot>`__


.. code-block:: console

   pip install "dm-meltingpot"


SMACv2
^^^^^^
:github:`null` `GitHub <https://github.com/oxwhirl/smacv2>`_


Follow the instructions on the environment `repository <https://github.com/oxwhirl/smacv2>`_.

`Here <https://github.com/facebookresearch/BenchMARL/blob/main/.github/unittest/install_smacv2.sh>`_
is how we install it on linux.

MAgent2
^^^^^^^
:github:`null` `GitHub <https://github.com/Farama-Foundation/MAgent>`__


.. code-block:: console

   pip install git+https://github.com/Farama-Foundation/MAgent2


Install models
--------------

Some models in BenchMARL require extra dependencies that can be installed separately


GNN
^^^

GNN models require :pyg:`null` `pytorch_geometric <https://pytorch-geometric.readthedocs.io/>`__.

To install it, you can run:

.. code-block:: console

   pip install torch_geometric

For more information, see the `installation <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`__ instructions of the library.
