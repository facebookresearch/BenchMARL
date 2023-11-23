
benchmarl.models
================

.. currentmodule:: benchmarl.models

.. contents:: Contents
    :local:

Here you can find the :ref:`model table <model-table>`.

Common
------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   Model
   ModelConfig
   SequenceModel
   SequenceModelConfig

Models
------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   {% for name in benchmarl.models.classes %}
     {{ name }}
   {% endfor %}
