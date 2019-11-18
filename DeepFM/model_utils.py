
import numpy as np
import tensorflow as tf
from itertools import combinations
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Adagrad
from DeepFM.data_utils import FEATURE_INFOS, InputKeys
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Optional, List


class DeepFM(object):
    def __init__(self, k: int, learning_rate: float, lambda_w: float = 2e-4, lambda_v: float = 2e-4):
        self.K = k
        self.lr = learning_rate
        self.lw = lambda_w
        self.lv = lambda_v
        self.w_regularizer = regularizers.l2(self.lw)
        self.v_regularizer = regularizers.l2(self.lv)
        self.inputs, linear_terms, v_embeddings = [], [], []

        for key, info in FEATURE_INFOS.items():
            inputs = layers.Input(shape=[info.length], name=key, dtype="int32")

            w_embedded = self.w_embedded(info.max_id + 1, inputs)  # [batch, 1]
            v_embedded = self.v_embedded(info.max_id + 1, inputs)   # [batch, K]

            self.inputs.append(inputs)
            linear_terms.append(w_embedded)
            v_embeddings.append(v_embedded)

        # FM part
        cross_terms = self.cross_dot(v_embeddings)
        self.y_1d = self.add(linear_terms)
        self.y_2d = self.add(cross_terms)

        # Deep part
        deep = layers.Concatenate(axis=1)(v_embeddings)  # [batch, K * p]
        for i in range(2):
            deep = layers.Dropout(rate=0.5)(deep)
            deep = layers.Dense(16, activation="relu")(deep)
        self.y_deep = layers.Dense(1, "relu", name="deep_output")(deep)

        # Output
        y = layers.Concatenate()([self.y_1d, self.y_2d, self.y_deep])
        y = layers.Dense(1, None, name="DeepFM_output")(y)

        self.model = Model(self.inputs, y)
        self.model.compile(
            optimizer=Adagrad(learning_rate=self.lr),  # optimizers with momentum won't work with embedding in keras
            loss="mse", metrics=["mae"]
        )

    def w_embedded(self, vocab_size: int, x: tf.Tensor):
        x = layers.Embedding(input_dim=vocab_size, output_dim=1,
                             embeddings_regularizer=self.w_regularizer,
                             mask_zero=True)(x)
        return layers.GlobalAvgPool1D()(x)

    def v_embedded(self, vocab_size: int, x: tf.Tensor):
        x = layers.Embedding(input_dim=vocab_size, output_dim=self.K,
                             embeddings_regularizer=self.v_regularizer,
                             mask_zero=True)(x)
        return layers.GlobalAvgPool1D()(x)  # multi-hot will be averaged

    @staticmethod
    def cross_dot(tensors: List[tf.Tensor]) -> List[tf.Tensor]:
        cross_values = []
        for f1, f2 in combinations(tensors, 2):
            dotted = layers.dot([f1, f2], axes=-1)
            cross_values.append(dotted)
        return cross_values

    @staticmethod
    def add(tensors: List[tf.Tensor]) -> tf.Tensor:
        if len(tensors) == 1:
            return tensors[0]
        return layers.Add()(tensors)

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

    def predict(self, users: np.ndarray, items: np.ndarray, occupations: np.ndarray, genres: np.ndarray) -> np.ndarray:
        return self.model.predict(
            {InputKeys.USER: users, InputKeys.ITEM: items,
             InputKeys.OCCUPATION: occupations,
             InputKeys.GENRES: genres})\
            .reshape(-1,)

    def save_model(self, file: str):
        self.model.save(file, overwrite=True)

    def load_model(self, file: str):
        self.model.load_weights(file)

    def summary(self):
        return self.model.summary()

    def plot_model(self, file: str):
        plot_model(self.model, file, show_shapes=True)



