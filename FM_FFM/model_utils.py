
import numpy as np
import tensorflow as tf
from itertools import combinations
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Adagrad, Adam
from FM_FFM.data_utils import InputKeys, FEATURE_MAX_NUM
from typing import Optional, List


class FM(object):
    def __init__(self, k: int, learning_rate: float = 0.2, lambda_w: float = 2e-4, lambda_v: float = 2e-4):
        self.K = k
        self.lr = learning_rate
        self.lw = lambda_w
        self.lv = lambda_v
        self.inputs = []

        const_input = layers.Input(shape=(1,), dtype="float32", name=f"{InputKeys.BIAS}")
        self.inputs.append(const_input)
        linear_terms = []
        cross_terms = []
        for name, n_features in FEATURE_MAX_NUM.items():
            inputs = layers.Input(shape=(1,), dtype="int32", name=f"{name}")

            # cross terms
            embedded = self.embedded(n_features, self.K, regularizers.l2(self.lv), inputs)

            # linear terms
            weighted = self.embedded(n_features, 1, regularizers.l2(self.lw), inputs)

            self.inputs.append(inputs)
            linear_terms.append(weighted)
            cross_terms.append(embedded)

        bias = layers.Dense(1, use_bias=False)(const_input)
        cross_terms = self.cross_dot(cross_terms)
        linear_values = layers.concatenate(linear_terms, axis=1)
        cross_values = layers.concatenate(cross_terms, axis=1) if len(cross_terms) > 1 else cross_terms[0]
        linear = layers.Lambda(self.mean)(linear_values)
        cross = layers.Lambda(self.mean)(cross_values)

        y = layers.Add()([bias, linear, cross])

        self._predict_func = K.function(self.inputs, outputs=y, name="pred_function")
        self.model = Model(inputs=self.inputs, outputs=y)
        self.model.compile(
            optimizer=Adagrad(learning_rate=self.lr),  # Adadelta will fill
            loss="mean_squared_error",
            metrics=["mae"]
        )

    @staticmethod
    def cross_dot(tensors: List[tf.Tensor]):
        cross_values = []
        for f1, f2 in combinations(tensors, 2):
            dotted = layers.dot([f1, f2], axes=1)
            cross_values.append(dotted)
        return cross_values

    @staticmethod
    def mean(tensors: List[tf.Tensor]):
        return K.sum(tensors, axis=1, keepdims=True)

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
            epochs=epochs,
            verbose=1,
            use_multiprocessing=False,
            workers=4,
            shuffle=False,
            callbacks=callbacks
        )
        return h

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return self._predict_func({InputKeys.USER: users,
                                   InputKeys.ITEM: items,
                                   InputKeys.BIAS: np.ones(shape=[len(users), 1], dtype=np.float32)})

    def save_model(self, file: str):
        self.model.save(file, overwrite=True)

    def load_model(self, file: str):
        self.model.load_weights(file)

    def summary(self):
        return self.model.summary()

    def plot_model(self, file: str):
        plot_model(self.model, file, show_shapes=True)
