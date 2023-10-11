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

    def on_batch_collected(self, batch: TensorDictBase):
        """
        A callback called at the end of every collection step.

        Args:
            batch (TensorDictBase): batch of collected data

        """
        pass

    def on_train_end(self, training_td: TensorDictBase):
        """
        A callback called at the end of every training step.

        Args:
            training_td (TensorDictBase): tensordict containing the loss values

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

    def on_batch_collected(self, batch: TensorDictBase):
        for callback in self.callbacks:
            callback.on_batch_collected(batch)

    def on_train_end(self, training_td: TensorDictBase):
        for callback in self.callbacks:
            callback.on_train_end(training_td)

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        for callback in self.callbacks:
            callback.on_evaluation_end(rollouts)
