Installation
============


Install TorchRL
-------------------------------

You can install :torchrl:`null` `TorchRL <https://github.com/pytorch/rl>`__ from PyPi.

.. code-block:: console

   pip install torchrl

For more details, or for installing nightly versions, see the
`TorchRL installation guide <https://github.com/pytorch/rl#installation>`__.

Install BenchMARL
-----------------

You can just install it from github

.. code-block:: console

   pip install benchmarl

Or also clone it locally to access the configs and scripts

.. code-block:: console

    git clone https://github.com/facebookresearch/BenchMARL.git
    pip install -e BenchMARL

Install environments
--------------------

All enviornment dependencies are optional in BenchMARL and can be installed separately.

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


SMACv2
^^^^^^
:github:`null` `GitHub <https://github.com/oxwhirl/smacv2>`_


Follow the instructions on the environment `repository <https://github.com/oxwhirl/smacv2>`_.

`Here <https://github.com/facebookresearch/BenchMARL/blob/main/.github/unittest/install_smacv2.sh>`_
is how we install it on linux.
