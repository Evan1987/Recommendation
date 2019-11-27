
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import History
from NCF.data_utils import InputKeys, FEATURE_MAX_INFOS
from _utils.utensorflow.model import KerasModel
from typing import Optional


class NCF(KerasModel):
    def __init__(self, k: int, learning_rate: float, dropout: float = 0.0):
        """
        Neural CF method
        :param k: The latent vector length for user and item
        :param learning_rate: The learning rate
        :param dropout: The dropout rate
        """
        self.K = k
        self.lr = learning_rate
        self.dropout = dropout
        self.inputs = []
        self.gmf = {}         # The gmf part
        self.mlp = {}         # The mlp part
        for key, max_id in FEATURE_MAX_INFOS.items():
            inputs = layers.Input(shape=(1,), name=key, dtype="int32")
            self.inputs.append(inputs)

            # same as embedding
            one_hotted = layers.Lambda(self.one_hot, arguments={"num_classes": max_id + 1}, input_shape=(1,))(inputs)
            gmf_dense = layers.Dense(self.K, activation="relu", name=f"{key}_gmf_dense")(one_hotted)
            self.gmf[key] = gmf_dense

            mlp_dense = layers.Dense(self.K, activation="relu", name=f"{key}_mlp_dense")(one_hotted)
            self.mlp[key] = mlp_dense

        # The GMF
        gmf = layers.Multiply(name="GMF")([self.gmf[InputKeys.USER], self.gmf[InputKeys.ITEM]])   # [batch, K]

        # The MLP
        interaction = layers.Concatenate(axis=-1, name="Interaction")(
            [self.mlp[InputKeys.USER], self.mlp[InputKeys.ITEM]])                               # [batch, 2 * K]

        mlp = self.dense(interaction, 2 * self.K, self.dropout)
        mlp = self.dense(mlp, self.K, self.dropout)
        mlp = self.dense(mlp, self.K // 2, self.dropout)                                        # [batch, K // 2]

        # Final output
        self.concatenation = layers.Concatenate(axis=-1, name="Concatenation")([gmf, mlp])      # [batch, K + K // 2]
        self.y = layers.Dense(1, activation="sigmoid")(self.concatenation)

        self._model = Model(inputs=self.inputs, outputs=self.y)
        self._model.compile(optimizer=optimizers.Adam(learning_rate=self.lr), loss="binary_crossentropy")

    @staticmethod
    def one_hot(x: tf.Tensor, num_classes: int):
        return K.one_hot(x, num_classes)

    @staticmethod
    def dense(x: tf.Tensor, units: int, rate: float):
        x = layers.Dense(units=units, activation="relu")(x)
        x = layers.Dropout(rate=rate)(x)
        return x

    @property
    def model(self):
        return self._model

    def train(self, train_data: Sequence, test_data: Optional[Sequence] = None,
              epochs: int = 1, callbacks=None) -> History:
        h = self._model.fit_generator(
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

    def predict(self, test_data, **kwargs):
        return self._model.predict(test_data)
