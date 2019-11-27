
"""
A User-defined Callback with `MRR`, `ReCall`, `DCG`
"""
import math
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
from NCF.data_utils import InputKeys, LABEL
from NCF.metrics import recall, mrr, normal_dcg

tqdm.pandas()


class MetricEvaluation(Callback):
    def __init__(self, val_data: pd.DataFrame, verbose: bool = True, topK: int = 3, predict_batch_size: int = 128):
        """Evaluate the model on val_data by multi metrics
        :param val_data: The validation data to evaluate
        :param verbose: Whether print
        :param topK: The num to recommend, e.g. to give the pred_items
        :param predict_batch_size: The batch size for model to predict once
        """
        super(MetricEvaluation, self).__init__()
        self.val_data = val_data
        self.verbose = verbose
        self.topK = topK
        self.predict_batch_size = predict_batch_size

    @staticmethod
    def metric_on_user(df: pd.DataFrame, topK: int):
        gt_item = df.loc[df[LABEL] == 1, InputKeys.ITEM].iloc[0]
        pred_items = df.sort_values(by="score", ascending=False)[InputKeys.ITEM][:topK].values
        recall_score, mrr_score, ndcg_score = [func(gt_item, pred_items) for func in [recall, mrr, normal_dcg]]
        return recall_score, mrr_score, ndcg_score

    def on_epoch_end(self, epoch, logs=None):
        # predict on all entries
        n = len(self.val_data)
        y_scores = np.empty((n,), dtype=np.float32)
        num_batches = math.ceil(n / self.predict_batch_size)
        for i in tqdm(range(num_batches)):
            s = slice(i * self.predict_batch_size, (i + 1) * self.predict_batch_size)
            batch_data = self.val_data.iloc[s]
            x = {key: batch_data[key].values.reshape(-1, 1) for key in [InputKeys.USER, InputKeys.ITEM]}
            pred = self.model.predict(x).reshape(-1)
            y_scores[s] = pred
        self.val_data["score"] = y_scores

        total_recall, total_mrr, total_ndcg = [], [], []
        for _, df in tqdm(self.val_data.groupby("user")):
            recall_score, mrr_score, ndcg_score = self.metric_on_user(df, self.topK)
            total_recall.append(recall_score)
            total_mrr.append(mrr_score)
            total_ndcg.append(ndcg_score)

        val_recall, val_mrr, val_ndcg = map(np.mean, [total_recall, total_mrr, total_ndcg])
        if self.verbose:
            print(f"Epoch: {epoch}, Recall: {val_recall:.4f}, MRR: {val_mrr:.4f}, NDCG: {val_ndcg:.4f}")
