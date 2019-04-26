# based on https://github.com/wuxiyu/transE

import os
import random
import numpy as np
import itertools as it
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import Optional, Set


class TransE(object):
    def __init__(self, *, triplets, entity_label_sep: Optional[str]=None, margin: float=1.0,
                 batch_size: int=64, learning_rate: float=1e-5, dim: int=10):
        """
        :param entity_label_sep: 标识 entity与其类型的分隔符，如果为 None，则表示没有注释类型
        :param triplets: 可迭代的三元组集合
        :param margin: HingeLoss的 margin
        :param learning_rate: 学习率
        :param dim: embedding的维度大小
        """
        self.sep = entity_label_sep
        self.default_ent_type = "__default_class__"  # 默认实体类型
        self.alpha = learning_rate
        self.gamma = margin
        self.batch_size = batch_size
        self.k = dim

        # 索引词典  id化
        self.ent2id = {}  # entity: (id, type)
        self.rel2id = {}  # relation: id

        # 反索引词典  反id化
        self.id2rel = {}  # id: (ent, type)
        self.id2ent = {}  # id: rel

        self.ent_index = 0  # 初始实体编号
        self.rel_index = 0  # 初始关系编号

        self.Triplet = namedtuple("Triplet", ["sub", "rel", "obj"])  # 其实定义这个没啥用，只是恰巧看见了感受下
        self.triplets: Set[self.Triplet] = set(self.collect(triplets))

        # 不同类型的实体集合 type: [id]
        self.ent_type_collection = {key: [id_ for id_, type_ in group]
                                    for key, group in it.groupby(self.ent2id.values(), key=lambda tup: tup[1])}

        self.graph = self._build_graph()
        self.sess = self._build_session()

    def add_ent_to_collection(self, ent, ent_type: str) -> int:
        """
        将目标实体元素收集起来
        :param ent: 实体
        :param ent_type: 实体类型
        :return: unit
        """
        if ent not in self.ent2id:
            self.ent2id[ent] = (self.ent_index, ent_type)
            self.id2ent[self.ent_index] = (ent, ent_type)
            self.ent_index += 1
            return self.ent_index - 1
        return self.ent2id[ent][0]

    def add_rel_to_collection(self, rel) -> int:
        """
        将目标关系收集起来
        :param rel: 关系
        :return: unit
        """
        if rel not in self.rel2id:
            self.rel2id[rel] = self.rel_index
            self.id2rel[self.rel_index] = rel
            self.rel_index += 1
            return self.rel_index - 1
        return self.rel2id[rel]

    def collect(self, triplets):
        """
        将给定的
        :param triplets:
        :return:
        """
        res = []
        for sub_, rel, obj_ in triplets:
            # 添加关系
            rel_id = self.add_rel_to_collection(rel)

            # 添加实体
            if self.sep is not None:
                sub, sub_ent_type = sub_.split(self.sep)
                sub_id = self.add_ent_to_collection(sub, sub_ent_type)

                obj, obj_ent_type = obj_.split(self.sep)
                obj_id = self.add_ent_to_collection(obj, obj_ent_type)
            else:
                # 为无类型标注的实体添加默认类型
                sub, obj = sub_, obj_
                sub_id = self.add_ent_to_collection(sub, self.default_ent_type)
                obj_id = self.add_ent_to_collection(obj, self.default_ent_type)

            triplet = self.Triplet(sub=sub_id, rel=rel_id, obj=obj_id)
            res.append(triplet)
        return res

    def _build_session(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=self.graph, config=config)
        return sess

    def _build_graph(self):
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope("embedding"):
                initializer = tf.random_uniform_initializer(minval=-6/(self.k ** 0.5), maxval=6/(self.k ** 0.5))
                self.embedding = tf.get_variable("embedding",
                                                 shape=[len(self.ent2id) + len(self.rel2id), self.k],
                                                 initializer=initializer)
            with tf.name_scope("input"):
                self.input = tf.placeholder(dtype=tf.int32, shape=[None, 6])  # :3为pos 3:为neg
                batch = tf.nn.embedding_lookup(self.embedding, self.input)  # shape: [batch_size, 3, k]
                batch /= tf.norm(batch, axis=-1, keep_dims=True)  # L2 normed

            with tf.name_scope("loss"):
                pos_loss = tf.reduce_sum(tf.squared_difference(batch[:, 0, :] + batch[:, 1, :], batch[:, 2, :]), axis=1)
                neg_loss = tf.reduce_sum(tf.squared_difference(batch[:, 3, :] + batch[:, 4, :], batch[:, 5, :]), axis=1)
                self.loss = tf.reduce_mean(tf.nn.relu(self.gamma + pos_loss - neg_loss))

            with tf.name_scope("opt"):
                self.train_op = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)

            self.init = tf.global_variables_initializer()
        return graph

    def get_batch(self):
        ent_num = len(self.ent2id)
        batch = []
        for sub, rel, obj in self.triplets:
            pos = [sub, rel + ent_num, obj]  # rel的索引叠加上ent的数量，方便embedding取数
            neg = pos.copy()
            # 通过随机数选择改变sub还是obj
            change_index = np.random.choice([0, 2])  # 0->sub, 2->obj
            _, type_ = self.id2ent[pos[change_index]]
            while True:
                temp = random.sample(self.ent_type_collection[type_], 1)[0]
                neg[change_index] = temp
                if tuple(neg) not in self.triplets:  # 此关系在样本中未出现，则生成该负样本
                    break
            batch.append(pos + neg)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def train(self, epochs: int=10000):
        self.sess.run(self.init)
        log = []
        for epoch in range(epochs):
            loss_collection = []
            batches = self.get_batch()
            for batch in batches:
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.input: batch})
                loss_collection.append(loss)
            avg_loss = np.mean(loss_collection)
            print("Epoch: %d, Avg.loss: %.4f" % (epoch, avg_loss))
            log.append(avg_loss)
        print("Done for Training!")
        plt.plot(log)

    def save(self, save_path: str):
        assert os.path.isdir(save_path), "%s is not a valid path" % save_path

        # save vectors
        vectors = self.sess.run(self.embedding)
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        ent_vectors = vectors[:len(self.ent2id)].astype(str)
        rel_vectors = vectors[len(self.ent2id):].astype(str)

        print("**** Writing Entities ****")
        with open(save_path + "ent.vec", "w") as f:
            f.write("id\tent\tvec\n")
            for i, vec in enumerate(ent_vectors):
                ent, _ = self.id2ent[i]
                f.write(str(i) + "\t" + ent + "\t" + (",".join(vec)))
                f.write("\n")

        print("**** Writing Relations ****")
        with open(save_path + "rel.vec", "w") as f:
            f.write("id\trel\tvec\n")
            for i, vec in enumerate(rel_vectors):
                rel = self.id2rel[i]
                f.write(str(i) + "\t" + rel + "\t" + (",".join(vec)))
                f.write("\n")

        print("Done!")


def read_file(file_path):
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                sub, obj, rel = line.split("\t")
                yield (sub, rel, obj)


if __name__ == '__main__':
    path = "F:/Code projects/Python/Recommendation/GraphEmbedding/test_data/WN18/"
    triplets = read_file(path + "train.txt")
    model = TransE(triplets=triplets, margin=1.0, learning_rate=1e-5, dim=10, batch_size=128)
    print("Num of Ent: %d,  Num of Rel: %d" % (len(model.ent2id), len(model.rel2id)))
    tf.summary.FileWriter("F:/board/TransE/", model.graph)
    model.train(15000)  # very slow
    model.save(path + "vec/")
