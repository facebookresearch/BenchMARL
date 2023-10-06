from __future__ import annotations

from typing import List

from tensordict import TensorDictBase


class Callback:
    def __init__(self):
        self.experiment = None

    def on_batch_collected(self, batch: TensorDictBase):
        pass

    def on_train_end(self, training_td: TensorDictBase):
        pass

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
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
