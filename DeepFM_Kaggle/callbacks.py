
"""
A User-defined Callback with `Gini`
"""
import numpy as np
from tensorflow.keras.callbacks import Callback
from DeepFM_Kaggle.metrics import gini
from typing import Tuple


class GiniCheckPoint(Callback):
    def __init__(self, filepath: str, min_delta=0, verbose=True, val_data: Tuple = None, predict_batch_size=None):
        """
        A custom callback based on `gini` score improvement and early-stopping.
        :param filepath: The dir path to save model
        :param min_delta: The threshold for min improvement.
        :param verbose: The controller for info print.
        :param val_data: The validation data for evaluate at the end of epoch.
        :param predict_batch_size: The size of validation data to evaluate on `gini`
        """
        super(GiniCheckPoint, self).__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.validation_data = val_data
        self.min_delta = min_delta
        self.predict_batch_size = predict_batch_size
        self.best = -np.inf  # The current best gini score
        self.monitor_op = np.greater  # The compare op to judge improvement

    def on_train_begin(self, logs=None):
        if self.verbose:
            print(f"Hi! Train Begin, logs: {logs}")
        self.best = -np.inf

    def on_train_end(self, logs=None):
        if self.verbose:
            print(f"Hi! Train finished, logs: {logs}")

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is not None:
            print("\nStart evaluate gini score on validation data.")
            X, y_true = self.validation_data
            y_scores = self.model.predict(X, batch_size=self.predict_batch_size)
            current = gini(y_true.reshape(-1), y_scores.reshape(-1))
            logs["val_gini"] = current
            logs["epoch"] = epoch

            if self.monitor_op(current - self.min_delta, self.best):
                print(f"Gini Score: {current:.4f} improved from {self.best:.4f}, saving model...")
                file = self.filepath.format(**logs)
                self.model.save_weights(file, overwrite=True)
                self.best = current
            else:
                print(f"Gini Score: {current:.4f} not improved from {self.best:.4f}, not save.")
