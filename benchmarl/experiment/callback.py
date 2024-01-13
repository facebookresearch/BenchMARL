#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import List

from tensordict import TensorDictBase


class Callback:
    """
    A Callback that can be added to experiments.
    To create your callback, you can inherit from this class
    and reimplement just the functions you need.

    Attributes:
        experiment (Experiment): the experiment associated to the callback
    """

    def __init__(self):
        self.experiment = None

    def on_setup(self):
        """A callback called atexperiment setup."""
        pass

    def on_batch_collected(self, batch: TensorDictBase):
        """
        A callback called at the end of every collection step.

        Args:
            batch (TensorDictBase): batch of collected data

        """
        pass

    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        """
        A callback called for every training step.

        Args:
           batch (TensorDictBase): tensordict with the training batch
           group (str): group name

        Returns:
            TensorDictBase: a new tensordict containing the loss values

        """
        pass

    def on_train_end(self, training_td: TensorDictBase, group: str):
        """
        A callback called at the end of training.

        Args:
            training_td (TensorDictBase): tensordict containing the loss values
            group (str): group name

        """
        pass

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        """
        A callback called at the end of every training step.

        Args:
            rollouts (list of TensorDictBase): tensordict containing the loss values

        """
        pass


class CallbackNotifier:
    def __init__(self, experiment, callbacks: List[Callback]):
        self.callbacks = callbacks
        for callback in self.callbacks:
            callback.experiment = experiment

    def _on_setup(self):
        for callback in self.callbacks:
            callback.on_setup()

    def _on_batch_collected(self, batch: TensorDictBase):
        for callback in self.callbacks:
            callback.on_batch_collected(batch)

    def _on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        train_td = None
        for callback in self.callbacks:
            td = callback.on_train_step(batch, group)
            if td is not None:
                if train_td is None:
                    train_td = td
                else:
                    train_td.update(td)
        return train_td

    def _on_train_end(self, training_td: TensorDictBase, group: str):
        for callback in self.callbacks:
            callback.on_train_end(training_td, group)

    def _on_evaluation_end(self, rollouts: List[TensorDictBase]):
        for callback in self.callbacks:
            callback.on_evaluation_end(rollouts)
