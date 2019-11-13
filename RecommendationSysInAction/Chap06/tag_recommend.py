"""6.5 Do recommendation based on items' tags. Using Last.fm DataSet."""

import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
import heapq
from tqdm import tqdm
from RecommendationSysInAction.utils.data import LastFM
from _utils.context import timer
from typing import Dict, Iterable, List, Tuple


logging.basicConfig(format='%(asctime)s [line: %(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger("logger")
PATH = r"F:\for learn\Python\推荐系统开发实战\章节代码\data\lastfm-2k"


class TagBasedRecommend(object):
    def __init__(self, k: float = 1.0):
        """
        Initialize Tag-recommendation
        :param k: The smooth parameter for computing user's favor on tag.
        """
        self.k = k

        self.max_tag_id = LastFM.max_tag_id
        self.max_artist_id = LastFM.max_artist_id
        self.artists = LastFM.load_artists_data()  # [id,name,url,pictureURL]
        self.total_artists = self.artists["id"].values
        self.user_rate_record = LastFM.load_rates_data()  # [userID,artistID,weight]
        self.user_tagged_record = LastFM.load_user_tagged_artists_data()[["userID", "artistID", "tagID"]]

        # shape: [max_artist_id + 1, max_tag_id + 1]
        self.artist_tag_gene: sp.csr_matrix =\
            self._generate_artist_tag(self.user_tagged_record, self.max_tag_id, self.max_artist_id)

        self.user_tag_preference: Dict[int, sp.csr_matrix] = self.generate_user_tag_preference()

        self.user_non_rated_artists: Dict[int, List[int]] =\
            self.generate_user_non_rated_artists(self.user_rate_record, self.total_artists)

    @staticmethod
    def _generate_artist_tag(artist_tagged_record: pd.DataFrame, max_tag_id: int, max_artist_id: int) -> sp.csr_matrix:
        """Generate sparse matrix for artists' tags, shape [max_artist_id + 1, max_tag_id + 1]
        :param artist_tagged_record: The total tagging record.
        :param max_tag_id: To help knowing the n_columns for matrix, each column's index indicates a tag_id.
        :param max_artist_id: To help knowing the n_rows for matrix, each row's index indicates a artist.
        :return: A sparse csr matrix of shape (max_artist_id + 1, max_tag_id + 1)
        """
        LOGGER.info(f"Generating Sparse tag matrix, shape: ({max_artist_id + 1}, {max_tag_id + 1}).")
        artist_tag_mapping: Dict[int, np.ndarray] = \
            artist_tagged_record.groupby("artistID").agg({"tagID": "unique"})["tagID"].to_dict()

        matrix = sp.dok_matrix((max_artist_id + 1, max_tag_id + 1), dtype=int)
        for artist_id, tags in tqdm(artist_tag_mapping.items(), desc="Generating Sparse tag matrix."):
            for tag_id in tags:
                matrix[artist_id, tag_id] = 1

        return matrix.tocsr()

    @staticmethod
    def _generate_user_tag_dependence(artist_tagged_record: pd.DataFrame, max_tag_id: int) -> Dict[int, sp.csr_matrix]:
        """
        Cal each user's dependence(tf-idf) on each tag, the method here is different from the one in original book.
        We consider each user is a 'document', the total tags tagged by same user are the `words` in document.
        and the `tf` and `idf` will be computed as follows:
            tf(user_u, tag_i) = N(tagging instances that user_u tagged tag_i) / N(total tagging instances by user_u)
            idf(tag_i) = log (N(total user) / (1 + N(user who ever tagged tag_i))
        :param artist_tagged_record: The total tagging record.
        :param max_tag_id: To help knowing the length for sparse row, each index indicates a tag_id.
        :return: A mapping between user and all tags' dependence which in sparse row.
        """
        LOGGER.info("Generating users' tag tf-idf dependency.")
        # user-tag-tf
        user_tag_summary = artist_tagged_record.groupby(["userID", "tagID"], as_index=False)\
            .agg({"artistID": "count"})\
            .rename(columns={"artistID": "tag_num"})
        user_tag_summary["total_tag_num"] = user_tag_summary.groupby("userID")["tag_num"].transform("sum")
        user_tag_summary["tag_tf"] = user_tag_summary["tag_num"] / user_tag_summary["total_tag_num"]

        # tag-idf
        n = artist_tagged_record["userID"].nunique()  # total tagging users
        tag_user_summary = artist_tagged_record.groupby("tagID", as_index=False)\
            .agg({"userID": "nunique"})\
            .rename(columns={"userID": "tag_num"})
        tag_user_summary["tag_idf"] = np.log(n / (tag_user_summary["tag_num"] + 1))

        # user-tag-tf_idf
        user_tag_tfidf = pd.merge(left=user_tag_summary[["userID", "tagID", "tag_tf"]],
                                  right=tag_user_summary[["tagID", "tag_idf"]],
                                  on="tagID")
        user_tag_tfidf["tag_tf_idf"] = user_tag_tfidf["tag_tf"] * user_tag_tfidf["tag_idf"]
        result = user_tag_tfidf[["userID", "tagID", "tag_tf_idf"]]\
            .groupby("userID")\
            .apply(
                # Generate a sparse row matrix for each user like coo_matrix's initialization
                # csr_matrix((data, (row, col)), shape, dtype)
                lambda df: sp.csr_matrix(
                    (df["tag_tf_idf"], (np.zeros(len(df["tag_tf_idf"])), df["tagID"])), shape=(1, max_tag_id + 1)
                )).to_dict()
        return result

    @staticmethod
    def _generate_user_rate(user_rate_record: pd.DataFrame, max_artist_id: int) -> Dict[int, sp.csr_matrix]:
        """Transform from original artist rate record to a mapping
        :param user_rate_record: Original artist rate record.
        :param max_artist_id: To help knowing the length for sparse row, each index indicates an artist_id.
        :return: mapping between user and it's rate on each artist.
        """
        LOGGER.info("Generate users' tag rate favor.")
        # scale down the number
        user_rate_record["weight"] /= 10000
        result = user_rate_record.groupby("userID").apply(
            lambda df: sp.csr_matrix(
                (df["weight"], (np.zeros(len(df["weight"])), df["artistID"])), shape=(1, max_artist_id + 1)
            )).to_dict()
        return result

    def generate_user_tag_preference(self) -> Dict[int, sp.csr_matrix]:
        """
        Compute user's total preference on tag for each user.
        The preference of user `u` on tag `t` can be computed as follows:
                Pre(u, t) = Rate(u, t) * TF_IDF(u, t)
            *)`TF_IDF(u, t)` is the user's dependence on tag.
            *)`Rate(u, t)` is the weighted mean of rate score on tag, which could be computed through score on artists:

                    Rate(u, t) = (sum_i(rate(u, i) * artist(i, t)) + r(u) * k) / (sum_i(artist(i, t)) + k)
                *) `rate(u, i)` is the user's rate score on artist `i`.
                *) `artist(i, t)` is the artist i's gene on tag `t`.
                *) `r(u)` is the mean rate score of user `u`.
                *) `k` is the smooth parameter.
        """
        LOGGER.info("Generate users' final preference for tag.")
        # Dict[int, csr_matrix(shape: [1, max_tag_id + 1])]
        user_tag_dependence = self._generate_user_tag_dependence(self.user_tagged_record, self.max_tag_id)

        # Dict[int, csr_matrix(shape: [1, max_artist_id + 1])]
        user_artist_rate = self._generate_user_rate(self.user_rate_record, self.max_artist_id)

        # shape: [max_artist_id + 1, max_tag_id + 1]
        artist_tag_gene = self.artist_tag_gene

        user_tag_preference: Dict[int, sp.csr_matrix] = {}
        for user, user_artist_rate in user_artist_rate.items():
            # Step1: generate user's rate for tag, using user's rate for artists and artist's tag gene.
            user_rate_mean = user_artist_rate.mean()  # r(u)
            user_rated_artists = user_artist_rate.sign()  # [1, max_artist_id + 1]

            user_tag_rate_total = user_artist_rate * artist_tag_gene  # shape: [1, max_tag_id + 1] -> main numerator
            tag_total = user_rated_artists * artist_tag_gene  # shape: [1, max_tag_id + 1] -> main denominator

            user_tag_rate = (user_tag_rate_total.toarray() + user_rate_mean * self.k) / (tag_total.toarray() + self.k)

            # Step2: generate total preference on tags
            # Element-wise multiply
            user_tag_preference[user] = user_tag_dependence[user].multiply(sp.csr_matrix(user_tag_rate))

        return user_tag_preference

    @staticmethod
    def generate_user_non_rated_artists(user_rate_record: pd.DataFrame, total_artists: Iterable[int]) -> Dict[int, List[int]]:

        LOGGER.info("Generate users' non-rated artists list.")
        if not isinstance(total_artists, set):
            total_artists = set(list(total_artists))

        user_non_rated_artists = user_rate_record.groupby("userID")\
            .agg({"artistID": lambda s: set(list(s))})\
            .pipe(lambda df: df["artistID"].apply(lambda s: sorted(list(total_artists - s))))\
            .to_dict()

        return user_non_rated_artists

    def recommend(self, user: int, n_items: int, recall_old: bool = False) -> List[Tuple[float, int]]:
        """Recommend `n_items` artists for user
        :param user: Target user.
        :param n_items: The num of recommendations.
        :param recall_old: Whether recall the artists that user ever rated, default `False`.
        """
        user_tag_preference = self.user_tag_preference[user]  # shape: [1, max_tag_id + 1]

        # make recall
        recall_artists = self.total_artists if recall_old else self.user_non_rated_artists[user]
        recall_artists_tag_gene = self.artist_tag_gene[recall_artists, :]  # shape: [#recall_artists, max_tag_id + 1]

        scores = user_tag_preference.dot(recall_artists_tag_gene.T).toarray().reshape(-1)  # shape: [#recall_artists, ]
        return heapq.nlargest(n_items, zip(recall_artists, scores), key=lambda pair: pair[1])

    def evaluate(self, user: int) -> Tuple[float, float]:
        """Evaluate recommendation on specific user.
        :return: Tuple of precision and recall
        """
        n_total_artists = len(self.total_artists)
        user_non_rated_artists = set(self.user_non_rated_artists[user])

        true_num = n_total_artists - len(user_non_rated_artists)
        pred = self.recommend(user, n_items=true_num, recall_old=True)

        hit = 0
        for artist, _ in pred:
            if artist not in user_non_rated_artists:
                hit += 1
        return hit / len(pred), hit / true_num


if __name__ == '__main__':
    rec = TagBasedRecommend(k=1.0)
    with timer("TagRecommend"):
        print(rec.recommend(2, 20, recall_old=False))  # 0.009s

    precision, recall = rec.evaluate(2)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")  # 0.2, 0.2
