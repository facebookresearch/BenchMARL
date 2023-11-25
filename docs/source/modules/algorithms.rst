
benchmarl.algorithms
====================

.. currentmodule:: benchmarl.algorithms

.. contents:: Contents
    :local:

Here you can find the :ref:`algorithm table <algorithm-table>`.

Common
------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class_private.rst

   Algorithm
   AlgorithmConfig

Algorithms
----------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class_private.rst

   {% for name in benchmarl.algorithms.classes %}
     {{ name }}
   {% endfor %}
