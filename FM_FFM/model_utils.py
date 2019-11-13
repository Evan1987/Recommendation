
import numpy as np
import tensorflow as tf
from itertools import combinations
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Adadelta
from FM_FFM.data_utils import InputKeys, FEATURE_MAX_NUM
from typing import Optional


class FM(object):
    def __init__(self, k: int, learning_rate: float = 0.01, lambda_w: float = 1e-3, lambda_v: float = 1e-3):
        self.K = k
        self.lr = learning_rate
        self.lw = lambda_w
        self.lv = lambda_v

        self.inputs = []
        bias = K.variable([0.0], dtype="float32", name="bias")
        linear_values = []
        cross_terms = []
        for name, n_features in FEATURE_MAX_NUM.items():
            inputs = layers.Input(shape=(1,), dtype="int32", name=f"{name}")

            # cross terms
            embedded = self.embedded(n_features, self.K, regularizers.l2(self.lv), inputs)

            # linear terms
            weighted = self.embedded(n_features, 1, regularizers.l2(self.lw), inputs)

            self.inputs.append(inputs)
            linear_values.append(weighted)
            cross_terms.append(embedded)

        cross_values = []
        for f1, f2 in combinations(cross_terms, 2):
            dotted = layers.dot([f1, f2], axes=1)
            cross_values.append(dotted)

        linear_values = layers.concatenate(linear_values, axis=1)
        cross_values = layers.concatenate(cross_values, axis=1) if len(cross_values) > 1 else cross_values[0]
        y = bias + K.sum(cross_values, axis=1, keepdims=True) + K.sum(linear_values, axis=1, keepdims=True)

        self._predict_func = K.function(self.inputs, outputs=y, name="pred_function")
        self.model = Model(inputs=self.inputs, outputs=y)
        self.model.compile(
            optimizer=Adadelta(learning_rate=self.lr),
            loss="mean_squared_error",
            metrics=["mae"]
        )

    @staticmethod
    def embedded(vocab_size: int, vec_length: int, embeddings_regularizer: regularizers.Regularizer,
                 x: tf.Tensor):
        e = layers.Embedding(input_dim=vocab_size, output_dim=vec_length,
                             input_length=1, embeddings_regularizer=embeddings_regularizer)
        x = layers.Flatten()(e(x))  # flatten not influence batch_size
        return x

    def train(self, train_data: Sequence, test_data: Optional[Sequence] = None,
              epochs: int = 1, callbacks=None) -> History:
        h = self.model.fit_generator(
            generator=train_data,
            validation_data=test_data,
            validation_steps=200,
            epochs=epochs,
            verbose=1,
            steps_per_epoch=len(train_data),
            use_multiprocessing=True,
            workers=4,
            callbacks=callbacks
        )
        return h

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return self._predict_func({InputKeys.USER: users, InputKeys.ITEM: items})

    def save_model(self, file: str):
        self.model.save(file, overwrite=True)

    def load_model(self, file: str):
        self.model.load_weights(file)

    def summary(self):
        return self.model.summary()

    def plot_model(self, file: str):
        plot_model(self.model, file, show_shapes=True)
