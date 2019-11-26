
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers, optimizers, regularizers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import History
from DeepFM_Kaggle.data_utils import FEAT_SIZE, NUMERIC_COLS, CATEGORICAL_COLS
from typing import Optional, List
from _utils.utensorflow.model import KerasModel


class GlobalSumPool(layers.Layer):
    """An implementation for sum pool"""
    def __init__(self, axis, **kwargs):
        super(GlobalSumPool, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layer
        return None

    def call(self, inputs, mask=None):
        if mask is None:
            if K.ndim(inputs) == 2:
                inputs = K.expand_dims(inputs)
            return K.sum(inputs, axis=self.axis)

        mask = K.cast(mask, "float32")
        # if K.ndim(mask) != K.ndim(inputs):
        #     mask = K.repeat(mask, inputs.shape[-1])
        #     mask = K.transpose(mask, [0, 2, 1])
        inputs = inputs * mask

        if K.ndim(inputs) == 2:
            inputs = K.expand_dims(inputs)
        return K.sum(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        out_shape = [input_shape[i] for i in range(len(input_shape)) if i != self.axis]
        if len(out_shape) == 1:
            out_shape.append(1)
        return tuple(out_shape)


class CrossProduct(layers.Layer):
    """An implementation for computing pair-wise dot-product for fm.
    The output not reduce sum to single scalar(shape: [batch, 1]) just for further compatibility.
    """
    def __init__(self, **kwargs):
        super(CrossProduct, self).__init__(**kwargs)
        self.supports_masking = False

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, **kwargs):
        left = K.sum(inputs, axis=1)                         # [batch, K]
        left = K.pow(left, 2)                                # [batch, K]

        right = K.pow(inputs, 2)                             # [batch, #len, K]
        right = K.sum(right, axis=1)                         # [batch, K]

        sub = left - right                                   # [batch, K]
        return 0.5 * sub                                     # [batch, K]

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0], input_shape[2]])


class PairWiseMultiply(layers.Layer):
    def __init__(self, **kwargs):
        super(PairWiseMultiply, self).__init__(**kwargs)
        self.supports_masking = False

    def call(self, inputs: tf.Tensor, **kwargs):
        _, n, m = inputs.shape.as_list()
        mask = self.build_mask(n)
        left = K.expand_dims(inputs, axis=2)                # [batch, n, 1, m]
        right = K.expand_dims(inputs, axis=1)               # [batch, 1, n, m]
        out = left * right                                  # [batch, n, n, m]
        return tf.boolean_mask(out, mask, axis=1)           # [batch, n * (n - 1) / 2, m]

    @staticmethod
    def build_mask(n: int):
        """Build the upper(with the diag) indices mask for n * n matrix"""
        ones = tf.ones(shape=(n, n), dtype="int32")
        upper = tf.matrix_band_part(ones, 0, -1)
        diag = tf.matrix_band_part(ones, 0, 0)
        return tf.cast(upper - diag, dtype="bool")          # [n, n]

    def compute_output_shape(self, input_shape):
        batch, n, m = input_shape
        return tuple([batch, n * (n - 1) // 2, m])


class Attention(layers.Layer):
    """Give the attention weight for each pair of cross-product features"""
    def __init__(self, units: int = 64, l2_reg: float = 0.01, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.l2_reg = l2_reg
        self.regularizer = regularizers.l2(l2_reg)
        self.supports_masking = False

    def build(self, input_shape):
        """Add variables
        :param input_shape: [batch, T, K], where `T = #FEAT * (#FEAT - 1) / 2` is the total interaction size. K is the
            embedding vector's length.
        """
        k = tf.TensorShape(input_shape).as_list()[-1]
        self.h = self.add_weight(name="attention_hidden", shape=[self.units, 1], initializer=tf.zeros_initializer())
        self.p = self.add_weight(name="pool_weight", shape=[k, 1], initializer=tf.zeros_initializer())
        self.dense = layers.Dense(units=self.units,
                                  activation="relu",
                                  use_bias=True,
                                  kernel_regularizer=self.regularizer,
                                  bias_regularizer=self.regularizer)

    def attention(self, inputs: tf.Tensor):
        """Calculate attention weights"""
        inner = self.dense(inputs)                                  # [batch, #FEAT * (#FEAT - 1) / 2, units]
        outer = tf.matmul(inner, self.h)                            # [batch, #FEAT * (#FEAT - 1) / 2, 1]
        return tf.nn.softmax(outer, axis=1)                         # [batch, #FEAT * (#FEAT - 1) / 2, 1]

    def call(self, inputs: tf.Tensor, **kwargs):
        """Make attention pooling
        :param inputs:   shape [batch, #FEAT * (#FEAT - 1) / 2, K] """
        att_w = self.attention(inputs)                              # [batch, #FEAT * (#FEAT - 1) / 2, 1]
        weighted_inputs = inputs * att_w                            # [batch, #FEAT * (#FEAT - 1) / 2, K]
        weighted_inputs = K.sum(weighted_inputs, axis=1)            # [batch, K]
        outputs = tf.matmul(weighted_inputs, self.p)                # [batch, 1]
        return outputs


class DeepFM(KerasModel):
    _valid_model_types = ["deepfm", "nfm", "afm"]

    def __init__(self, learning_rate: float = 1e-3, k: int = 8, model_type: str = "deepfm",
                 use_2d: bool = None, use_1d: bool = True, final_dnn: bool = False):
        if model_type not in self._valid_model_types:
            raise ValueError(f"Unknown model type `{model_type}`, possible choices are {self._valid_model_types}")
        self.K = k
        self.lr = learning_rate
        self.model_type = model_type
        self.use_2d = use_2d
        self.use_1d = use_1d
        if self.use_2d is None and self.model_type in ["nfm", "afm"]:
            self.use_2d = False

        self.final_dnn = final_dnn
        self.optimizer = optimizers.Adam(self.lr)

        self.inputs, embed_cols = [], []

        numeric_cols = []
        for col in NUMERIC_COLS:
            input_ = layers.Input(shape=(1,), name=col)  # [batch, 1]
            self.inputs.append(input_)
            embedded = layers.RepeatVector(1)(layers.Dense(self.K)(input_))   # [batch, 1, K] -> for 2d special embed
            embed_cols.append(embedded)
            numeric_cols.append(input_)  # collect feat values
        con_numeric = layers.Concatenate(axis=1)(numeric_cols)                # [batch, #NUMERIC_COLS]
        dense_numeric = layers.RepeatVector(1)(layers.Dense(1)(con_numeric))  # [batch, 1, 1] -> for 1d

        categorical_cols = []
        for col in CATEGORICAL_COLS:
            input_ = layers.Input(shape=(1,), name=col)  # [batch, 1]
            self.inputs.append(input_)
            vocab_size = FEAT_SIZE[col]
            embedded_1 = layers.Embedding(vocab_size, 1)(input_)              # [batch, 1, 1] -> for 1d
            embedded_2 = layers.Embedding(vocab_size, self.K)(input_)         # [batch, 1, K] -> for 2d
            embed_cols.append(embedded_2)
            categorical_cols.append(embedded_1)  # !Different from numeric, collect feat_value * weight
        dense_categorical = layers.Concatenate(axis=1)(categorical_cols)      # [batch, #CATEGORICAL_COLS, 1] -> for 1d

        # 1-order
        self.y_1d = layers.Concatenate(axis=1)([dense_numeric, dense_categorical])  # [batch, #FEAT, 1]
        self.y_1d = GlobalSumPool(axis=1)(self.y_1d)                                # [batch, 1] -> 1d for last layer

        # 2-order
        emb = layers.Concatenate(axis=1)(embed_cols)                          # [batch, #FEAT, K]

        if self.model_type == "deepfm":
            self.y = self.deep_fm(self.y_1d, emb)
        elif self.model_type == "nfm":
            self.y = self.nfm(self.y_1d, emb)
        elif self.model_type == "afm":
            self.y = self.afm(self.y_1d, emb)

        self._model = Model(inputs=self.inputs, outputs=self.y)
        self._model.compile(optimizer=self.optimizer, loss="binary_crossentropy")

    @staticmethod
    def dnn(x: tf.Tensor):
        """Deep part for common deep-fm hidden layers
        :return: Tensor in [batch, 1] shape
        """
        for _ in range(2):
            x = layers.Dense(32, activation="relu")(x)
            x = layers.Dropout(0.5)(x)

        x = layers.Dense(1, activation="relu")(x)
        return layers.Dropout(0.5)(x)

    def final_output(self, tensors: List[tf.Tensor]):
        """Give the final output for last layer of concat tensors."""
        if len(tensors) > 1:
            y = layers.Concatenate()(tensors)
        else:
            y = tensors[0]
        if self.final_dnn:
            return layers.Dense(1, activation="sigmoid")(y)
        return layers.Softmax()(GlobalSumPool(axis=1)(y))

    def deep_fm(self, y_1d: tf.Tensor, embeddings: tf.Tensor):
        """Treat all inputs in deep-fm method
        :param y_1d:            shape [batch, 1]
        :param embeddings:      shape [batch, #FEAT, K]
        :return: final output   shape [batch, 1]
        """
        y_2d = GlobalSumPool(axis=1)(CrossProduct()(embeddings))        # [batch, 1]
        y_deep = layers.Flatten()(embeddings)                           # [batch, #FEAT * K]
        y_deep = self.dnn(y_deep)                                       # [batch, 1]
        last_layer = []
        if self.use_1d:
            last_layer.append(y_1d)
        if self.use_2d:
            last_layer.append(y_2d)
        last_layer.append(y_deep)
        return self.final_output(last_layer)

    def nfm(self, y_1d: tf.Tensor, embeddings: tf.Tensor):
        """Treat all inputs in nfm method
        :param y_1d:            shape [batch, 1]
        :param embeddings:      shape [batch, #FEAT, K]
        :return: final output   shape [batch, 1]
        """
        y_2d = CrossProduct()(embeddings)                               # [batch, K]
        y_deep = self.dnn(y_2d)                                         # [batch, 1]
        last_layer = []
        if self.use_1d:
            last_layer.append(y_1d)
        if self.use_2d:
            last_layer.append(y_2d)
        last_layer.append(y_deep)
        return self.final_output(last_layer)

    def afm(self, y_1d: tf.Tensor, embeddings: tf.Tensor):
        """Treat all inputs in afm method
        :param y_1d:            shape [batch, 1]
        :param embeddings:      shape [batch, #FEAT, K]
        :return: final output   shape [batch, 1]
        """
        y_2d = PairWiseMultiply()(embeddings)                           # [batch, #FEAT * (#FEAT - 1) / 2, K]
        y_deep = Attention(64)(y_2d)                                    # [batch, 1]
        last_layer = []
        if self.use_1d:
            last_layer.append(y_1d)
        last_layer.append(y_deep)
        return self.final_output(last_layer)

    @property
    def model(self):
        return self._model

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

    def predict(self, test_data, **kwargs):
        return self.model.predict(test_data)
