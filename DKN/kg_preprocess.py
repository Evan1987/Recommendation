"""
读取图谱 TransX后生成的向量数据：
    -  Trans{$X}_entity2vec_{$dim}.vec      entity_index(kg) -> ent_vec
    -  Trans{$X}_relation2vec_{$dim}.vec    relation_index(kg) -> rel_vec
结合index映射：
    - entity2id.txt      entity_id -> entity_index(kg)  图谱index
    - relation2id.txt    relation_id -> relation_index(kg)
    - triple2id.txt      encoded by index(kg) for each row(head rel tail)
    - entity2index.txt   entity_id -> entity_index(data)  业务数据 index（与图谱 index不同）
根据实际业务数据，生成可用的 embedding数据
"""

import os
import numpy as np
from collections import defaultdict
from typing import Dict, List
from DKN.constant import PATH


class Embedding4Data(object):

    entity2index_file = "entity2index.txt"
    entity2id_file = "entity2id.txt"
    triplet2id_file = "triple2id.txt"
    relation2id_file = "relation2id.txt"

    def __init__(self, *, base_path: str):
        self.base_path = base_path
        self.entity2index = self.read_map(self.entity2index_file)
        self.entity2id = self.read_map(self.entity2id_file)
        self.entity2neighbours = self.get_neighbour_for_entity(self.triplet2id_file)

    def read_map(self, file) -> Dict[str, int]:
        """
        读入mapping文件，并生成映射字典
        id -> index
        """
        m = {}
        with open(os.path.join(self.base_path, file), "r", encoding="utf-8") as f:
            for line in f:
                array = line.split("\t")
                if len(array) != 2:  # to skip first line in entity2id.txt
                    continue
                obj, index = array[0], int(array[1])
                m[obj] = index
        return m

    def get_neighbour_for_entity(self, file) -> Dict[int, List[int]]:
        """
        读入三元组文件，生成mapping：实体及与其直接相连的实体列表
        entity_index(kg) -> [entity_indexes(kg)]
        """
        entity2neighbour_map = defaultdict(list)
        with open(os.path.join(self.base_path, file), "r", encoding="utf-8") as f:
            for line in f:
                array = line.split("\t")
                if len(array) != 3:  # to skip first line in triple2id.txt
                    continue
                head_index, tail_index, _ = map(int, array)
                entity2neighbour_map[head_index].append(tail_index)
                entity2neighbour_map[tail_index].append(head_index)
        return dict(entity2neighbour_map)

    def embedding_transform(self, entity_embedding_file: str):
        """
        基于业务数据生成合适的 embedding矩阵
        entity_embeddings: entity_index(data) -> ent_vec  # key换了
        context_embeddings: entity_index(data) -> neighbour_vec (average pooling of neighbours' ent_vec)
        :param entity_embedding_file: transX输出的 entity embedding 文件.
        """
        kge_method = entity_embedding_file.split("_")[0]
        full_entity_embeddings = np.loadtxt(os.path.join(self.base_path, entity_embedding_file))
        dim = full_entity_embeddings.shape[1]

        entity_embeddings, context_embeddings = [np.zeros(shape=[len(self.entity2index) + 1, dim]) for _ in range(2)]
        for entity_id, index in self.entity2index.items():
            if entity_id in self.entity2id:
                kg_index = self.entity2id[entity_id]
                entity_embeddings[index] = full_entity_embeddings[kg_index]
                if kg_index in self.entity2neighbours:
                    neighbour_indices = self.entity2neighbours[kg_index]
                    context_embeddings[index] = np.mean(full_entity_embeddings[neighbour_indices], axis=0)

        np.save(os.path.join(self.base_path, "entity_embeddings_%s_%d" % (kge_method, dim)), entity_embeddings)
        np.save(os.path.join(self.base_path, "context_embeddings_%s_%d" % (kge_method, dim)), context_embeddings)


if __name__ == '__main__':
    base_path = os.path.join(PATH, "kg")
    preprocessor = Embedding4Data(base_path=base_path)
    entity_embedding_file = [file for file in os.listdir(base_path)
                             if file.split(".")[-1] == "vec" and "entity2vec" in file][0]

    print("Start Generating embeddings")
    preprocessor.embedding_transform(entity_embedding_file)
