
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
from sklearn.metrics import roc_auc_score
from typing import List
from collections import namedtuple

BATCH = namedtuple("Batch", ["clicked_words", "clicked_entities", "news_words", "news_entities", "labels"])


class DKN(object):
    def __init__(self, *, word_emb_file: str, ent_emb_file: str, context_emb_file: str=None,
                 max_title_length: int=10, max_click_history: int=30, out_channels: int=128,
                 filter_sizes: List[int]=[1, 2], l2_weight: float=0.01, lr: float=0.001,
                 use_context: bool=True, transform: bool=True, save_graph: bool=True):
        """
        DKN模型
        :param word_emb_file: 存放 word2vec路径
        :param ent_emb_file: 存放 ent2vec的路径
        :param context_emb_file: 存放 context2vec的路径
        :param max_title_length: 最大标题词数
        :param max_click_history:  最大历史点击数
        :param out_channels: 卷积操作的输出通道数
        :param filter_sizes: 卷积操作的不同卷积核大小
        :param l2_weight: l2 loss的权重
        :param lr: 学习率
        :param use_context: 是否使用context特征
        :param transform: 是否在做embedding前，对embedding做对等转换
        """

        self.word_embs, self.ent_embs = np.load(word_emb_file), np.load(ent_emb_file)
        self.max_title_length = max_title_length
        self.max_click_history = max_click_history
        self.out_channels = out_channels
        self.filter_sizes = filter_sizes
        self.use_context = use_context
        self.context_embs = np.load(context_emb_file) if use_context else None
        self.transform = transform
        self.entity_dim = self.ent_embs.shape[1]
        self.word_dim = self.word_embs.shape[1]
        self.l2_weight = l2_weight
        self.lr = lr
        self.params = []
        self.graph = self._build_graph()
        if save_graph:
            tf.summary.FileWriter("F:/board/DKN/", self.graph)

    def _kcnn(self, words: tf.Tensor, entities: tf.Tensor):
        """
        卷积操作函数，数据平行输入不同卷积核的卷积层，最后将结果拼接
        Input words and entities must have same shape with `[-1, max_title_length]`.
        Especially, 
        clicked_words and clicked_entities have shape `[batch_size * max_click_history, max_title_length]`
        while news_words and news_entities have shape `[batch_size, max_title_length]`
        :return: tensor output from single layer of convs (include max_pooling)
                 with shape: `[-1, out_channels * n_filters]`
                 E.P. clicked: [batch_size * max_click_history, out_channels * n_filters]
                      news:    [batch_size, out_channels * n_filters]
        """

        # [batch_size * max_click_history, max_title_length, word_dim] for clicked_words
        # [batch_size, max_title_length, word_dim] for new words
        embedded_words = tf.nn.embedding_lookup(self.word_embeddings, words)

        # [batch_size * max_click_history, max_title_length, entity_dim] for clicked_entities
        # [batch_size, max_title_length, entity_dim] for new entities
        embedded_entities = tf.nn.embedding_lookup(self.entity_embeddings, entities)

        # [batch_size * max_click_history, max_title_length, full_dim] for clicked
        # [batch_size, max_title_length, full_dim] for news
        if self.use_context:
            embedded_contexts = tf.nn.embedding_lookup(self.context_embeddings, entities)
            concat_input = tf.concat([embedded_words, embedded_entities, embedded_contexts], axis=-1)
            full_dim = self.word_dim + 2 * self.entity_dim
        else:
            concat_input = tf.concat([embedded_words, embedded_entities], axis=-1)
            full_dim = self.word_dim + self.entity_dim

        # [batch_size * max_click_history, max_title_length, full_dim, 1] for clicked
        # [batch_size, max_title_length, full_dim, 1] for news
        # input must be of shape: [batch, in_height, in_width, in_channels]
        concat_input = tf.expand_dims(concat_input, axis=-1)  # add channel: in_channel=1

        outputs = []
        for filter_size in self.filter_sizes:

            # filter must be of shape: [filter_height, filter_width, in_channels, out_channels]
            filter_shape = [filter_size, full_dim, 1, self.out_channels]
            w = tf.get_variable(name="w_" + str(filter_size), shape=filter_shape, dtype=tf.float32)
            b = tf.get_variable(name="b_" + str(filter_size), shape=[self.out_channels, ], dtype=tf.float32)
            if w not in self.params:
                self.params.append(w)

            # [batch_size * max_click_history, max_title_length - filter_size + 1, 1, out_channels] for clicked
            # [batch_size, max_title_length - filter_size + 1, 1, out_channels] for news
            conv = tf.nn.conv2d(input=concat_input, filter=w, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            relu = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            # [batch_size * max_click_history, 1, 1, out_channels] for clicked
            # [batch_size, 1, 1, out_channels] for news
            pool = tf.nn.max_pool(relu, ksize=[1, self.max_title_length - filter_size + 1, 1, 1],
                                  strides=[1, 1, 1, 1], padding="VALID", name="pool")
            outputs.append(pool)

        # [batch_size * max_click_history, 1, 1, out_channels * len(filter_sizes)] for clicked
        # [batch_size, 1, 1, out_channels * len(filter_sizes)] for news
        output = tf.concat(outputs, axis=-1)

        # [batch_size * max_click_history, out_channels * len(filter_sizes)] for clicked
        # [batch_size, out_channels * len(filter_sizes)] for news
        # output = tf.reshape(output, shape=[-1, self.out_channels * len(self.filter_sizes)])
        return tf.squeeze(output, axis=[1, 2])

    def _build_graph(self):
        graph = tf.Graph()
        tf.reset_default_graph()
        with graph.as_default():
            with tf.name_scope("input"):
                self.clicked_words = tf.placeholder(dtype=tf.int32, shape=[None, self.max_click_history, self.max_title_length], name="clicked_words")
                self.clicked_entities = tf.placeholder(dtype=tf.int32, shape=[None, self.max_click_history, self.max_title_length], name="clicked_entities")
                self.news_words = tf.placeholder(dtype=tf.int32, shape=[None, self.max_title_length], name="news_words")
                self.news_entities = tf.placeholder(dtype=tf.int32, shape=[None, self.max_title_length], name="news_entities")
                self.labels = tf.placeholder(dtype=tf.float32, shape=[None, ], name="labels")

            with tf.name_scope("embedding"):
                self.word_embeddings = tf.Variable(self.word_embs, dtype=tf.float32, name="word")
                self.entity_embeddings = tf.Variable(self.ent_embs, dtype=tf.float32, name="entity")
                self.params.append(self.word_embeddings)
                self.params.append(self.entity_embeddings)
                if self.use_context:
                    self.context_embeddings = tf.Variable(self.context_embs, dtype=tf.float32, name="context")
                    self.params.append(self.context_embeddings)

                if self.transform:
                    self.entity_embeddings = tf.layers.dense(inputs=self.entity_embeddings, units=self.entity_dim,
                                                             activation=tf.nn.tanh, name="transformed_entity",
                                                             kernel_regularizer=l2_regularizer(self.l2_weight))
                    if self.use_context:
                        self.context_embeddings = tf.layers.dense(inputs=self.context_embeddings, units=self.entity_dim,
                                                                  activation=tf.nn.tanh, name="transformed_context",
                                                                  kernel_regularizer=l2_regularizer(self.l2_weight))

            with tf.name_scope("attention"):
                # [batch_size * max_click_history, max_title_length]
                clicked_words = tf.reshape(self.clicked_words, shape=[-1, self.max_title_length])
                clicked_entities = tf.reshape(self.clicked_entities, shape=[-1, self.max_title_length])

                with tf.variable_scope("kcnn", reuse=tf.AUTO_REUSE):
                    # title_embedding_length =out_channels * n_filters

                    # [batch_size * max_click_history, title_embedding_length]
                    clicked_embeddings = self._kcnn(clicked_words, clicked_entities)

                    # [batch_size, title_embedding_length]
                    news_embeddings = self._kcnn(self.news_words, self.news_entities)

                # [batch_size, max_click_history, title_embedding_length]
                clicked_embeddings = tf.reshape(clicked_embeddings,
                                                shape=[-1, self.max_click_history, self.out_channels * len(self.filter_sizes)])

                # [batch_size, 1, title_embedding_length]
                news_embeddings_expanded = tf.expand_dims(news_embeddings, axis=1)

                # [batch_size, max_click_history]
                attention_weights = tf.reduce_sum(clicked_embeddings * news_embeddings_expanded, axis=-1)

                # [batch_size, max_click_history]
                attention_weights = tf.nn.softmax(attention_weights, dim=-1)

                # [batch_size, max_click_history, 1]
                attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)

                # [batch_size, title_embedding_length]
                user_embeddings = tf.reduce_sum(clicked_embeddings * attention_weights_expanded, axis=1)

            with tf.name_scope("train"):
                # [batch_size, ]
                self.logits = tf.reduce_sum(user_embeddings * news_embeddings, axis=1)

                self.base_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
                )
                self.l2_loss = tf.Variable(0., dtype=tf.float32)
                for param in self.params:
                    self.l2_loss += self.l2_weight * tf.nn.l2_loss(param)
                if self.transform:
                    self.l2_loss += tf.losses.get_regularization_loss()

                self.loss = self.base_loss + self.l2_loss
                self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            self.init = tf.global_variables_initializer()
        return graph

    def _extract_batch(self, batch: BATCH):
        return {
            self.clicked_entities: batch.clicked_entities,
            self.clicked_words: batch.clicked_words,
            self.news_entities: batch.news_entities,
            self.news_words: batch.news_words,
            self.labels: batch.labels
        }

    def train(self, sess: tf.Session, batch: BATCH):
        feed_dict = self._extract_batch(batch)
        return sess.run(self.optimizer, feed_dict=feed_dict)

    def eval(self, sess: tf.Session, batch: BATCH):
        feed_dict = self._extract_batch(batch)
        scores = sess.run(self.logits, feed_dict=feed_dict)
        return roc_auc_score(y_true=batch.labels, y_score=scores)




