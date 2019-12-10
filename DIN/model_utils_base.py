
import tensorflow as tf
import numpy as np
from .data_utils import InputKeys
from typing import Dict


def dice(x: tf.Tensor, name: str = ""):
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable(name=f"alpha{name}", shape=x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.), dtype=tf.float32)
        beta = tf.get_variable(name=f"beta{name}", shape=x.get_shape()[-1],
                               initializer=tf.constant_initializer(0.), dtype=tf.float32)

    # a simple way to use BN to calculate x_p
    x_normed = tf.layers.batch_normalization(x, center=False, scale=False, name=name, reuse=tf.AUTO_REUSE)
    x_p = tf.sigmoid(beta * x_normed)
    return alphas * x * (1.0 - x_p) + x_p * x


class DeepInterestNet(object):
    hidden_units = 128

    def __init__(self, user_count: int, item_count: int, cate_count: int):
        self.user_count = user_count
        self.item_count = item_count
        self.cate_count = cate_count
        self.graph = tf.Graph()
        tf.reset_default_graph()
        with self.graph.as_default():
            with tf.variable_scope("inputs"):
                self.user = tf.placeholder(dtype=tf.int32, shape=(None, ), name="user")  # not used
                self.item = tf.placeholder(dtype=tf.int32, shape=(None, ), name="item")
                self.cate = tf.placeholder(dtype=tf.int32, shape=(None, ), name="cate")
                self.label = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="label")
                self.hist_item = tf.placeholder(dtype=tf.int32, shape=(None, None), name="hist_item")
                self.hist_cate = tf.placeholder(dtype=tf.int32, shape=(None, None), name="hist_cate")
                self.hist_length = tf.placeholder(dtype=tf.int32, shape=(None, ), name="sequence_length")
                self.lr = tf.placeholder(dtype=tf.float64, shape=None, name="learning_rate")

            with tf.variable_scope("embedding"):
                user_emb_w = tf.get_variable(name="user_emb_w", shape=[user_count, self.hidden_units])  # not used
                item_emb_w = tf.get_variable(name="item_emb_w", shape=[item_count, self.hidden_units // 2])
                item_emb_b = tf.get_variable(name="item_emb_b", shape=[item_count, 1],
                                             initializer=tf.constant_initializer(0.))
                cate_emb_w = tf.get_variable(name="cate_emb_w", shape=[cate_count, self.hidden_units // 2])

            with tf.variable_scope("embedded"):
                with tf.variable_scope("user_emb"):
                    user_emb = tf.nn.embedding_lookup(user_emb_w, self.user)            # [B, H], not used
                with tf.variable_scope("item_emb"):
                    # concat item embedding and its cate's embedding
                    item_emb = tf.concat([
                        tf.nn.embedding_lookup(item_emb_w, self.item),                  # [B, H/2], embed for item
                        tf.nn.embedding_lookup(cate_emb_w, self.cate)                   # [B, H/2], embed for cate
                    ], axis=-1)                                                         # [B, H]
                    item_emb_b = tf.nn.embedding_lookup(item_emb_b, self.item)          # [B, 1]
                with tf.variable_scope("history_emb"):
                    # concat history items embedding and their cate's embedding
                    hist_emb = tf.concat([
                        tf.nn.embedding_lookup(item_emb_w, self.hist_item),             # [B, T, H/2]
                        tf.nn.embedding_lookup(cate_emb_w, self.hist_cate)              # [B, T, H/2]
                    ], axis=-1)                                                         # [B, T, H]

            with tf.variable_scope("history_attention"):
                print(item_emb.shape)
                print(hist_emb.shape)
                hist = self.attention(item_emb, hist_emb, self.hist_length)             # [B, 1, H]

            with tf.variable_scope("fc_input"):
                hist = tf.layers.batch_normalization(hist)
                hist = tf.squeeze(hist, axis=1)                                         # [B, H]
                hist = tf.layers.dense(hist, units=self.hidden_units, name="hist_fcn")  # [B, H]

            with tf.variable_scope("fc"):
                x = tf.concat([hist, item_emb], axis=-1, name="x")                      # [B, 2 * H]
                x = tf.layers.batch_normalization(x, name="bn_x")
                x = tf.layers.dense(x, units=80, activation=None, name="f1")
                x = dice(x, name="dice_f1")
                x = tf.layers.dense(x, units=40, activation=None, name="f2")
                x = dice(x, name="dice_f2")
                x = tf.layers.dense(x, units=1, activation=None, name="f3")             # [B, 1]
                self.logits = item_emb_b + x                                            # [B, 1]

            with tf.variable_scope("opt"):
                self.global_step = tf.Variable(0, trainable=False, name="global_step")
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits, labels=self.label
                ))
                trainable_params = tf.trainable_variables()
                self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
                self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params),
                                                         global_step=self.global_step)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    @staticmethod
    def attention(queries: tf.Tensor, keys: tf.Tensor, keys_length: tf.Tensor) -> tf.Tensor:
        """
        DIN attention part
        :param queries: shape [B, H], the target item embedding
        :param keys: shape [B, T, H], the history item embeddings
        :param keys_length: shape [B], the true length of history
        :return: tensor with shape [B, 1, H]
        """
        t = tf.shape(keys)[1]
        h = keys.get_shape().as_list()[-1]
        queries = tf.tile(tf.expand_dims(queries, axis=1), [1, t, 1])   # [B, T, H], copy H for T times for each sample
        din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)       # [B, T, 4 * H]

        dnn = tf.layers.dense(din_all, units=80, activation=tf.nn.sigmoid, name="att_f1")
        dnn = tf.layers.dense(dnn, units=40, activation=tf.nn.sigmoid, name="att_f2")
        dnn = tf.layers.dense(dnn, units=1, activation=None, name="att_f3")                 # [B, T, 1]
        outputs = tf.transpose(dnn, [0, 2, 1])                                              # [B, 1, T]

        key_masks = tf.sequence_mask(keys_length, maxlen=t)                                 # [B, T], bool tensor
        key_masks = tf.expand_dims(key_masks, axis=1)                                       # [B, 1, T]
        padding = tf.ones_like(outputs) * (-2 ** 32 + 1)                                    # [B, 1, T]

        # Pad with small values, not zero
        outputs = tf.where(key_masks, outputs, padding)                                     # [B, 1, T]
        outputs = outputs / (h ** 0.5)                                                      # [B, 1, T], scale
        outputs = tf.nn.softmax(outputs)                                                    # [B, 1, T], activation
        return tf.matmul(outputs, keys)                                                     # [B, 1, H]

    def train(self, sess: tf.Session, feed: Dict[str, np.ndarray], lr: float):
        loss, _ = sess.run([self.loss, self.train_op],
                           feed_dict={
                               self.user: feed[InputKeys.USER],
                               self.item: feed[InputKeys.ITEM],
                               self.cate: feed[InputKeys.CATE],
                               self.label: feed[InputKeys.LABEL],
                               self.hist_item: feed[InputKeys.HISTORY_ITEM],
                               self.hist_cate: feed[InputKeys.HISTORY_CATE],
                               self.hist_length: feed[InputKeys.HISTORY_LENGTH],
                               self.lr: lr
                           })
        return loss

    def test(self, sess: tf.Session, feed: Dict[str, np.ndarray]) -> np.ndarray:
        score = sess.run(self.logits,
                         feed_dict={
                             self.user: feed[InputKeys.USER],
                             self.item: feed[InputKeys.ITEM],
                             self.cate: feed[InputKeys.CATE],
                             self.hist_item: feed[InputKeys.HISTORY_ITEM],
                             self.hist_cate: feed[InputKeys.HISTORY_CATE],
                             self.hist_length: feed[InputKeys.HISTORY_LENGTH],
                         })
        return score

    def save(self, sess: tf.Session, file: str):
        self.saver.save(sess, save_path=file)

    def restore(self, sess: tf.Session, file: str):
        self.saver.restore(sess, file)
