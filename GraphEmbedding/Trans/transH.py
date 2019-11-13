# based on https://github.com/thunlp/TensorFlow-TransX
# according to http://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/download/8531/8546

import os
import random
import numpy as np
import itertools as it
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Set


class TransH(object):
    def __init__(self, *, triplets, entity_label_sep: Optional[str]=None, margin: float=1.0, batch_size: int=64,
                 learning_rate: float=0.01, dim: int=50, C: float=0.25, epsilon: float=1e-4):
        """
        :param entity_label_sep: 标识 entity与其类型的分隔符，如果为 None，则表示没有注释类型
        :param triplets: 可迭代的三元组集合
        :param margin: HingeLoss的 margin
        :param learning_rate: 学习率
        :param dim: embedding的维度大小
        :param C: 损失函数中限制条件损失的超参
        :param epsilon: 限制条件损失中的正交损失超参
        """
        self.sep = entity_label_sep
        self.default_ent_type = "__default_class__"  # 默认实体类型
        self.alpha = learning_rate
        self.gamma = margin
        self.batch_size = batch_size
        self.k = dim
        self.C = C
        self.epsilon = epsilon

        # 索引词典  id化
        self.ent2id = {}  # entity: (id, type)
        self.rel2id = {}  # relation: id

        # 反索引词典  反id化
        self.id2rel = {}  # id: (ent, type)
        self.id2ent = {}  # id: rel

        self.ent_index = 0  # 初始实体编号
        self.rel_index = 0  # 初始关系编号

        # 计算各个关系的tph(#{tail entities} per head), pth(#{head entities} per tail)
        # rel: (tph, pth)
        self.rel_mapping_summary = {}

        self.triplets: Set[Tuple[int]] = set(self.collect(triplets))

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

            res.append((sub_id, rel_id, obj_id))

        # 统计每种关系的tph和pth
        group = it.groupby(res, key=lambda tup: tup[1])
        for rel, iters in group:
            subs, objs = [], []
            for sub, _, obj in iters:
                subs.append(sub)
                objs.append(obj)
            tph = len(set(objs)) / len(set(subs))
            pth = 1 / tph
            self.rel_mapping_summary[rel] = (tph, pth)

        return res

    def _build_session(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=self.graph, config=config)
        return sess

    def calc(self, e, w):
        w = tf.nn.l2_normalize(w, dim=-1)
        return e - tf.reduce_sum(e * w, axis=1, keep_dims=True) * w

    def _build_graph(self):
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope("input"):
                self.pos_sub = tf.placeholder(dtype=tf.int32, shape=[None], name="pos_sub")
                self.pos_rel = tf.placeholder(dtype=tf.int32, shape=[None], name="pos_rel")
                self.pos_obj = tf.placeholder(dtype=tf.int32, shape=[None], name="pos_obj")

                self.neg_sub = tf.placeholder(dtype=tf.int32, shape=[None], name="neg_sub")
                self.neg_rel = tf.placeholder(dtype=tf.int32, shape=[None], name="neg_rel")
                self.neg_obj = tf.placeholder(dtype=tf.int32, shape=[None], name="neg_obj")

            with tf.name_scope("embedding"):
                initializer = tf.contrib.layers.xavier_initializer(uniform=False)
                self.e_embedding = tf.get_variable("e_embedding",
                                                   shape=[len(self.ent2id), self.k],
                                                   initializer=initializer)
                self.r_embedding = tf.get_variable("r_embedding",
                                                   shape=[len(self.rel2id), self.k],
                                                   initializer=initializer)
                self.w_embedding = tf.get_variable("w_embedding",
                                                   shape=[len(self.rel2id), self.k],
                                                   initializer=initializer)

                pos_sub_e = tf.nn.embedding_lookup(self.e_embedding, self.pos_sub, name="pos_sub_e")  # [None, k]
                pos_rel_e = tf.nn.embedding_lookup(self.r_embedding, self.pos_rel, name="pos_rel_e")  # [None, k]
                pos_obj_e = tf.nn.embedding_lookup(self.e_embedding, self.pos_obj, name="pos_obj_e")  # [None, k]

                neg_sub_e = tf.nn.embedding_lookup(self.e_embedding, self.neg_sub, name="neg_sub_e")  # [None, k]
                neg_rel_e = tf.nn.embedding_lookup(self.r_embedding, self.neg_rel, name="neg_rel_e")  # [None, k]
                neg_obj_e = tf.nn.embedding_lookup(self.e_embedding, self.neg_obj, name="neg_obj_e")  # [None, k]

                pos_w = tf.nn.embedding_lookup(self.w_embedding, self.pos_rel, name="pos_w")  # [None, k]
                neg_w = tf.nn.embedding_lookup(self.w_embedding, self.neg_rel, name="neg_w")  # [None, k]

                pos_sub_e, pos_obj_e = [self.calc(e, pos_w) for e in [pos_sub_e, pos_obj_e]]
                neg_sub_e, neg_obj_e = [self.calc(e, neg_w) for e in [neg_sub_e, neg_obj_e]]

            with tf.name_scope("loss"):
                with tf.name_scope("hinge_loss"):
                    pos_loss = tf.reduce_sum(tf.squared_difference(pos_sub_e + pos_rel_e, pos_obj_e), axis=1)
                    neg_loss = tf.reduce_sum(tf.squared_difference(neg_sub_e + neg_rel_e, neg_obj_e), axis=1)
                    hinge_loss = tf.reduce_mean(tf.nn.relu(self.gamma + pos_loss - neg_loss))

                with tf.name_scope("scale_loss"):
                    # sum_e [|e|^2 - 1]+
                    all_e = tf.concat([pos_sub_e, pos_obj_e, neg_sub_e, neg_obj_e], axis=0)
                    scale_loss = tf.reduce_mean(tf.nn.relu(tf.norm(all_e, axis=1) - 1.0), name="scale_loss")

                with tf.name_scope("orthogonal_loss"):
                    # sum_r [<wr, dr>^2 / |dr|^2 - epsilon]+
                    all_w = tf.concat([pos_w, neg_w], axis=0)
                    all_r = tf.concat([pos_rel_e, neg_rel_e], axis=0)
                    normed_w = tf.nn.l2_normalize(all_w, dim=-1)
                    inner_product = tf.reduce_sum(normed_w * all_r, axis=1)
                    l2_r = tf.norm(all_r, axis=1)
                    orthogonal_loss = tf.reduce_mean(tf.nn.relu(inner_product ** 2 / l2_r - self.epsilon))

                self.loss = hinge_loss + self.C * (scale_loss + orthogonal_loss)

            with tf.name_scope("opt"):
                self.train_op = tf.train.GradientDescentOptimizer(self.alpha).minimize(self.loss)

            self.init = tf.global_variables_initializer()
        return graph

    def get_batch(self):
        batch = []
        for sub, rel, obj in self.triplets:
            pos = [sub, rel, obj]
            neg = pos.copy()

            # 通过加权随机选择改变sub还是obj，1vN的多给予sub机会，Nv1的多给予obj机会
            tph, pth = self.rel_mapping_summary[rel]
            change_index = np.random.choice([0, 2], p=[tph / (tph + pth), pth / (tph + pth)])  # 0->sub, 2->obj
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
                batch = np.asarray(batch)
                feed_dict = {}
                for i, input_ in enumerate([self.pos_sub, self.pos_rel, self.pos_obj,
                                            self.neg_sub, self.neg_rel, self.neg_obj]):
                    feed_dict[input_] = batch[:, i]
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                loss_collection.append(loss)
            avg_loss = np.mean(loss_collection)
            print("Epoch: %d, Avg.loss: %.4f" % (epoch, avg_loss))
            log.append(avg_loss)
        print("Done for Training!")
        plt.plot(log)

    def save(self, save_path: str):
        assert os.path.isdir(save_path), "%s is not a valid path" % save_path

        # save vectors
        ent_vectors = self.sess.run(self.e_embedding).astype(str)
        rel_vectors = self.sess.run(self.r_embedding).astype(str)
        w_vectors = self.sess.run(self.w_embedding).astype(str)
        w_vectors /= np.linalg.norm(w_vectors, axis=-1, keepdims=True)

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

        print("**** Writing Hyperplane Normal Vectors ****")
        with open(save_path + "hyper.vec", "w") as f:
            f.write("id\trel\tvec\n")
            for i, vec in enumerate(w_vectors):
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
    model = TransH(triplets=triplets, margin=1.0, learning_rate=0.01, dim=10, batch_size=64)
    print("Num of Ent: %d,  Num of Rel: %d" % (len(model.ent2id), len(model.rel2id)))
    tf.summary.FileWriter("F:/board/TransH/", model.graph)
    model.train(15000)  # very slow
    model.save(path + "vec/")
