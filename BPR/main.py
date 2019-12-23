
# Ref http://ethen8181.github.io/machine-learning/recsys/4_bpr.html

from BPR.data_utils import make_data
from BPR.model_utils import BayesianPersonalizedRanking


pos_threshold = 0.3
test_size = 0.2
seed = 2019
k = 15
reg = 0.01
learning_rate = 0.1
batch_size = 100
epochs = 160


train_data, test_data, neg_collections = make_data(pos_threshold, test_size, seed)


if __name__ == '__main__':
    bpr = BayesianPersonalizedRanking(learning_rate, k, reg, random_state=seed)
    bpr.fit(train_data, batch_size, epochs)
    acc_collection, auc_collection = bpr.evaluate(test_data, neg_collections)
    print(f"Acc: {sum(acc_collection) / len(acc_collection)}, Auc: {sum(auc_collection) / len(auc_collection)}")
    # Acc: 0.813606145761885, Auc: 0.8141153155211344
