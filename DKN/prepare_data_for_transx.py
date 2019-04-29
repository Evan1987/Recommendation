"""
处理图谱原始数据，为Trans系列算法准备输入数据
"""
import os
from typing import Optional, Dict
from DKN.constant import PATH


class KG4Trans(object):
    def __init__(self, *, kg_file: str, output_path: Optional[str]=None):
        """
        读取原始kg数据，输出
        :param kg_file: 数据路径，记录 triplets三元组 head[TAB]relation[TAB]tail
        :param output_path:
        """
        self.output_path = output_path if output_path else os.getcwd()
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.kg_file = kg_file if os.path.isfile(kg_file) else os.path.join(self.output_path, kg_file)
        self.entity2id: Dict[str, int] = {}  # entity_id: index
        self.relation2id: Dict[str, int] = {}  # rel_id: index

    @staticmethod
    def _add_to_collection(obj: str, collection: Dict[str, int], index: int):
        """
        将目标obj加入collection中，并返回其映射序号和当前待添加序号
        :param obj: 待加入元素
        :param collection: 收集集合
        :param index: 待加入的序号
        :return: obj对应的序号，加入obj后当前待加入序号
        """
        if obj not in collection:
            link_index = index
            collection[obj] = index
            index += 1
        else:
            link_index = collection[obj]
        return link_index, index

    def read(self):
        ent_cnt, rel_cnt = 0, 0
        count = 0
        with open(self.kg_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    head, rel, tail = line.split("\t")
                    head_index, ent_cnt = self._add_to_collection(head, self.entity2id, ent_cnt)
                    tail_index, ent_cnt = self._add_to_collection(tail, self.entity2id, ent_cnt)
                    rel_index, rel_cnt = self._add_to_collection(rel, self.relation2id, rel_cnt)
                    count += 1
                    yield (head_index, tail_index, rel_index)


    def extract(self):
        ent_cnt, rel_cnt = 0, 0
        triplet2id_file = os.path.join(self.output_path, "triplet2id.txt")
        entity2id_file = os.path.join(self.output_path, "entity2id.txt")
        relation2id_file = os.path.join(self.output_path, "relation2id.txt")

        # encoding triplet and collect entity & relation to dict
        print("Saving encoded triplets")
        with open(self.kg_file, "r", encoding="utf-8") as fr, \
                open(triplet2id_file, "w", encoding="utf-8") as fw:


            for line in fr:
                line = line.strip()
                if line:
                    head, rel, tail = line.split("\t")
                    head_index, ent_cnt = self._add_to_collection(head, self.entity2id, ent_cnt)
                    tail_index, ent_cnt = self._add_to_collection(tail, self.entity2id, ent_cnt)
                    rel_index, rel_cnt = self._add_to_collection(rel, self.relation2id, rel_cnt)
                    fw.write("%d\t%d\t%d\n" % (head_index, tail_index, rel_index))
                    count += 1
            fw.seek(0, 0)
            fw.write("%d" % count)

        # 写入entity2id
        print("Saving entities")
        with open(entity2id_file, "w", encoding="utf-8") as f:
            f.write("%d\n" % len(self.entity2id))
            for ent_id, index in self.entity2id.items():
                f.write("%s\t%d\n" % (ent_id, index))

        # 写入relation2id
        print("Saving relations")
        with open(relation2id_file, "w", encoding="utf-8") as f:
            f.write("%d\n" % len(self.relation2id))
            for rel_id, index in self.relation2id.items():
                f.write("%s\t%d\n" % (rel_id, index))

        print("Succeed in extracting KG data. All done!")


if __name__ == '__main__':
    kg_path = os.path.join(PATH, "kg")
    kg_file = os.path.join(kg_path, "kg.txt")
    preprocessor = KG4Trans(kg_file=kg_file, output_path=kg_path)
    preprocessor.extract()


