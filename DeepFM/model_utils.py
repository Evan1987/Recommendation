
import numpy as np
import tensorflow as tf
from itertools import combinations
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Adagrad, Adam
from DeepFM.data_utils import FEATURE_MAX, InputKeys
from typing import Optional, List


class DeepFM(object):
    def __init__(self, k: int, learning_rate: float, lambda_w: float = 2e-4, lambda_v: float = 2e-4):
        self.K = k
        self.lr = learning_rate
        self.lw = lambda_w
        self.lv = lambda_v
        self.w_regularizer = regularizers.l2(self.lw)
        self.v_regularizer = regularizers.l2(self.lv)
        self.inputs, linear_terms, cross_terms = [], [], []

        # For one-hot input
        for key, max_ in FEATURE_MAX.items():
            if key == InputKeys.GENRES:  # special
                continue
            inputs = layers.Input(shape=[1], name=key, dtype="int32")
            # cross terms
            cross_embedded = self.embedded(max_ + 1, self.K, self.v_regularizer, inputs)
            # linear terms
            linear_embedded = self.embedded(max_ + 1, 1, self.w_regularizer, inputs)

            self.inputs.append(inputs)
            linear_terms.append(linear_embedded)
            cross_terms.append(cross_embedded)

        # For multi-hot input
        max_ = FEATURE_MAX[InputKeys.GENRES]
        genres = layers.Input(shape=[max_ + 1], sparse=True, name=InputKeys.GENRES, dtype="int32")
        cross_embedded = layers.Embedding(input_dim=max_ + 1, output_dim=self.K, input_length=max_ + 1,
                                          embeddings_regularizer=self.v_regularizer)
        cross_embedded


    @staticmethod
    def embedded(vocab_size: int, vec_length: int, embeddings_regularizer: regularizers.Regularizer,
                 x: tf.Tensor):
        e = layers.Embedding(input_dim=vocab_size, output_dim=vec_length,
                             input_length=1, embeddings_regularizer=embeddings_regularizer)
        x = layers.Flatten()(e(x))  # flatten not influence batch_size
        return x
