
import os
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict
from DIN.data_utils import DataGenerator, dataset, cate_mapping, PACKAGE_HOME, InputKeys
from DIN.model_utils_base import DeepInterestNet
from _utils.utensorflow import get_session_config


BATCH_SIZE = 32
TEST_BATCH_SIZE = 128
SEED = 1234
EPOCHS = 50
LR = 1.0
MODEL_HOME = os.path.join(PACKAGE_HOME, "model")
os.makedirs(MODEL_HOME, exist_ok=True)
train_data = dataset.query("is_train == 1")
test_data = dataset.query("is_train == 0")


def epoch_test(target_model: DeepInterestNet, seq: DataGenerator, session: tf.Session):
    n_test_batches = len(seq)
    logs = defaultdict(dict)
    for j in tqdm(range(n_test_batches), desc="Test on model"):
        test_batch = seq[j]
        scores = target_model.test(session, test_batch).reshape(-1)
        for user, label, score in zip(test_batch[InputKeys.USER], test_batch[InputKeys.LABEL], scores):
            logs[user][label] = score

    auc = sum([label_scores[1] > label_scores[0]
               for label_scores in logs.values() if len(label_scores) == 2]) / len(logs)
    return auc


if __name__ == '__main__':
    train_seq = DataGenerator(train_data, cate_mapping, BATCH_SIZE, SEED)
    test_seq = DataGenerator(test_data, cate_mapping, TEST_BATCH_SIZE, SEED)
    n_batches = len(train_seq)
    model = DeepInterestNet(train_seq.user_count, train_seq.item_count, train_seq.cate_count)
    with tf.Session(graph=model.graph, config=get_session_config()) as sess:
        sess.run(model.init)
        lr = LR
        for epoch in range(EPOCHS):
            loss = 0
            for i in tqdm(range(n_batches)):
                batch = train_seq[i]
                loss += model.train(sess, batch, lr)
            test_auc = epoch_test(model, test_seq, sess)
            print(f"Epoch {epoch + 1} loss: {loss / n_batches:.4f}, test auc: {test_auc:.4f}")

            train_seq.on_epoch_end()

