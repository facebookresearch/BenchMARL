Reporting and plotting
======================

Reporting and plotting is compatible with `marl-eval <https://github.com/instadeepai/marl-eval>`__.
If ``experiment.create_json=True`` (this is the default in the `experiment config <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/experiment/base_experiment.yaml>`__)
a file named ``{experiment_name}.json`` will be created in the experiment output folder with the format of `marl-eval <https://github.com/instadeepai/marl-eval>`__.
You can load and merge these files using the utils in `eval_results <https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/eval_results.py>`__
to create beautiful plots of
your benchmarks.  No more struggling with matplotlib and latex!

.. python_example_button::
   https://github.com/facebookresearch/BenchMARL/blob/main/examples/plotting/plot_benchmark.py

Example plots
-------------

Here are some example plots you can generate, for more info, check out `marl-eval <https://github.com/instadeepai/marl-eval>`__.


Aggregate scores
^^^^^^^^^^^^^^^^

.. figure:: https://raw.githubusercontent.com/matteobettini/benchmarl_sphinx_theme/master/benchmarl_sphinx_theme/static/img/benchmarks/vmas/aggregate_scores.png
   :align: center

   Aggregate scores

Sample efficiency curves
^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: https://raw.githubusercontent.com/matteobettini/benchmarl_sphinx_theme/master/benchmarl_sphinx_theme/static/img/benchmarks/vmas/environemnt_sample_efficiency_curves.png
   :align: center

   Sample efficiency curves


Performance profile
^^^^^^^^^^^^^^^^^^^

.. figure:: https://raw.githubusercontent.com/matteobettini/benchmarl_sphinx_theme/master/benchmarl_sphinx_theme/static/img/benchmarks/vmas/performance_profile_figure.png
   :align: center

   Performance profile
