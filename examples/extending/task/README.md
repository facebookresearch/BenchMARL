# Creating a new task

In the following we will see how to:
1. Create new tasks from a new environment
2. Create new tasks from an existing environment



## Creating new tasks from a new environment

Here are the steps to create a new task and a new environment. 

1. Create your `CustomEnvTask` and `CustomEnvClass` following the example in [`environments/customenv/common.py`](environments/customenv/common.py).
This is an enum with task entries and a class with abstract functions you need to implement. The entries of the enum will be the 
uppercase names of your tasks.
2. Create a `conf/task/customenv` folder with a yaml configuration file for each of your tasks. This folder will have a 
yaml configuration for each task. You can see [`conf/task/customenv`](conf/task/customenv) for an example. (**NOTE:** only include the `defaults` section at the top of the file if you are doing step 6)
3. Place your task script in [`benchmarl/environments/customenv/common.py`](../../../benchmarl/environments) and 
your config in [`benchmarl/conf/task/customenv`](../../../benchmarl/conf/task) (or any other place you want to 
override from).
4. Add `CustomEnvTask` to [`benchmarl.environments.tasks`](../../../benchmarl/environments/__init__.py) list.
5. Load it with
```bash
python benchmarl/run.py task=customenv/task_1 algorithm=...
```

6. (Optional) You can create python dataclasses to use as schemas for your tasks
to validate their config. This will allow to check the configuration entries and types for each task.
This step is optional and, if you skip it, everything will work (just without the task config being checked against python dataclasses).
To do it, just create `environments/customenv/taskname.py` for each task, with a `TaskConfig` object following the structure shown in 
[`environments/customenv/task_1.py`](environments/customenv/task_1.py). In our example, `task_1` has such dataclass, while `task_2`
doesn't. The name of the python file has to be the name of the task in lower case. Then you need to tell hydra to use 
this as a schema by adding `customenv_taskname_config` to the defaults at the top of the task yaml file.
See [`conf/task/customenv/task_1.yaml`](conf/task/customenv/task_1.yaml) for an example.

## Creating new tasks from an existing environment

Imagine we now have already in the library `customenv` with `task_1` and `task_2`.
To create a new task (e.g., `task_3`) in an existing environment , follow these steps:

1. Add `TASK_3 = None` to `CustomEnvTask` in [`environments/customenv/common.py`](environments/customenv/common.py).
2. Add `task_3.yaml` to [`conf/task/customenv`](conf/task/customenv)

3. (Optional) Add `task_3.py` to [`environments/customenv`](environments/customenv) and 
the default `customenv_task_3_config` at the top of `task_3.yaml`.


## PettingZoo example

PR [#84](https://github.com/facebookresearch/BenchMARL/pull/84) contains an example on how to add a your own PettingZoo task.
