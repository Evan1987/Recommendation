
"""
A User-defined Callback with `Gini`
"""
import numpy as np
from tensorflow.keras.callbacks import Callback
from DeepFM_Kaggle.metrics import gini


class GiniWithEarlyStopping(Callback):
    def __init__(self, min_delta=0, patience=0, verbose=0, predict_batch_size: int = 1024):
        super(GiniWithEarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.best = -np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size = predict_batch_size

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        if self.verbose > 1:
            print(f"Hi! Train Begin, logs: {logs}")
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.inf

    def on_train_end(self, logs=None):
        if self.verbose > 1:
            print(f"Hi! Train finished, logs: {logs}")

        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Epoch {self.stopped_epoch} gini early stopping!")

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data:
            X, y = self.validation_data
            y_scores = self.model.predict(self.validation_data[0], batch_size=self.predict_batch_size)
            gini_score = gini(self.validation_data[1], y_scores)


