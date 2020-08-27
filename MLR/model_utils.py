
from tensorflow.keras import layers, optimizers, models,activations
from evan_utils.utensorflow.model import KerasModel


class MLR(KerasModel):
    def __init__(self, d: int, m: int, learning_rate: float):
        self.d = d
        self.m = m
        self.lr = learning_rate
        self.opt = optimizers.Adam(self.lr)

        self.inputs = layers.Input(shape=(self.d, ), name="inputs", dtype="float32")
        partition_pred = layers.Dense(units=self.m, use_bias=False, activation="softmax")(self.inputs)    # [batch, m]
        label_pred = layers.Dense(units=self.m, use_bias=False, activation="sigmoid")(self.inputs)        # [batch, m]
        self.y_pred = layers.Dot(axes=1)([partition_pred, label_pred])                                    # [batch, 1]
        self._model = models.Model(inputs=self.inputs, outputs=self.y_pred)
        self._model.compile(optimizer=self.opt, loss="binary_crossentropy")

    @property
    def model(self):
        return self._model

    def train(self, train_data, test_data=None, epochs: int = 10, callbacks=None):
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

    def predict(self, test_data, **kwargs):
        return self.model.predict(test_data)

