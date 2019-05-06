# Based on https://github.com/princewen/tensorflow_practice/tree/master/recommendation/Basic-DKN-Demo
# According to https://arxiv.org/abs/1801.08284v1

import os
import tensorflow as tf
from DKN.data_loader import TreatingData, get_batch, transform_to_batch
from DKN.model import DKN
from DKN.constant import PATH
from DKN.data_preprocess import MAX_TITLE_LENGTH

max_click_history = 30
max_title_length = MAX_TITLE_LENGTH
out_channels = 128
filter_sizes = [1, 2]
l2_weight = 0.01
lr = 0.001
batch_size = 128
epochs = 10


def get_embedding_file(data_path: str, kg_path: str):
    """在路径中查找所需文件"""
    word_embedding_file = [file for file in os.listdir(data_path) if file.split(".")[-1] == "npy"][0]
    word_embedding_file = os.path.join(data_path, word_embedding_file)

    entity_embedding_file = [file for file in os.listdir(kg_path)
                             if file.split(".")[-1] == "npy" and "entity" in file][0]
    entity_embedding_file = os.path.join(kg_path, entity_embedding_file)

    context_embedding_file = [file for file in os.listdir(kg_path)
                              if file.split(".")[-1] == "npy" and "context" in file][0]
    context_embedding_file = os.path.join(kg_path, context_embedding_file)

    return word_embedding_file, entity_embedding_file, context_embedding_file


def build_session(graph):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu90%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.Session(graph=graph, config=config)
    return sess


if __name__ == '__main__':
    data_path = os.path.join(PATH, "data")
    kg_path = os.path.join(PATH, "kg")

    '''Embedding 文件路径'''
    word_emb_file, ent_emb_file, context_emb_file = get_embedding_file(data_path, kg_path)

    '''处理数据'''
    train_file, test_file = [os.path.join(data_path, x) for x in ["train.txt", "test.txt"]]
    treater = TreatingData(train_file=train_file, test_file=test_file, max_click_history=max_click_history)
    treater.transform()
    train, test = treater.train, treater.test


    model = DKN(word_emb_file=word_emb_file, ent_emb_file=ent_emb_file, context_emb_file=context_emb_file,
                max_title_length=max_title_length, max_click_history=max_click_history, out_channels=out_channels,
                filter_sizes=filter_sizes, l2_weight=l2_weight, lr=lr, use_context=True, transform=True, save_graph=True)

    with build_session(model.graph) as sess:
        sess.run(model.init)

        for epoch in range(epochs):
            for batch in get_batch(train, batch_size):
                model.train(sess, batch)

            # train_auc = model.eval(sess, transform_to_batch(train))
            test_auc = model.eval(sess, transform_to_batch(test))
            print("Epoch: %d\tTest Auc: %.4f" % (epoch, test_auc))
