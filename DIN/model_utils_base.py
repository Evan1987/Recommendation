
import tensorflow as tf
from typing import List


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


class Model(object):
    hidden_units = 128

    def __init__(self, user_count: int, item_count: int, cate_count: int, cate_list: List[int]):
        self.user_count = user_count
        self.item_count = item_count
        self.cate_count = cate_count
        self.cate_list = cate_list
        self.graph = tf.Graph()
        tf.reset_default_graph()
        with self.graph.as_default():
            self.cate_list_mapping = tf.convert_to_tensor(self.cate_list, dtype=tf.int64)

            with tf.variable_scope("inputs"):
                self.user = tf.placeholder(dtype=tf.int32, shape=(None, ), name="user")
                self.item_i = tf.placeholder(dtype=tf.int32, shape=(None, ), name="item_i")
                self.item_j = tf.placeholder(dtype=tf.int32, shape=(None, ), name="item_j")
                self.label = tf.placeholder(dtype=tf.int32, shape=(None, ), name="label")
                self.hist_i = tf.placeholder(dtype=tf.int32, shape=[None, None], name="hist_i")
                self.hist_length = tf.placeholder(dtype=tf.int32, shape=(None, ), name="sequence_length")
                self.lr = tf.placeholder(dtype=tf.float64, shape=None, name="learning_rate")

            with tf.variable_scope("embedding"):
                user_emb_w = tf.get_variable(name="user_emb_w", shape=[user_count, self.hidden_units])
                item_emb_w = tf.get_variable(name="item_emb_w", shape=[item_count, self.hidden_units // 2])
                item_b = tf.get_variable(name="item_b", shape=[item_count], initializer=tf.constant_initializer(0.))
                cate_emb_w = tf.get_variable(name="cate_emb_w", shape=[cate_count, self.hidden_units // 2])


