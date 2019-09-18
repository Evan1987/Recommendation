"""6.5 Do recommendation based on items' tags. Using Last.fm DataSet."""

import os
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
from RecommendationSysInAction.utils.data import LastFM
from typing import Dict


logging.basicConfig(format='%(asctime)s [line: %(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger("logger")
PATH = r"F:\for learn\Python\推荐系统开发实战\章节代码\data\lastfm-2k"


class TagBasedRecommend(object):
    def __init__(self):
        self.artists = LastFM.load_artists_data()  # [id,name,url,pictureURL]
        self.user_rate_record = LastFM.load_rates_data()  # [userID,artistID,weight]
        self.user_tagged_record = LastFM.load_user_tagged_artists_data()[["userID", "artistID", "tagID"]]
        self.artist_tags = self.generate_artist_tag(self.user_tagged_record, LastFM.max_tag_id, LastFM.max_artist_id)

    @staticmethod
    def generate_artist_tag(artist_tagged_record: pd.DataFrame, max_tag_id: int, max_artist_id: int) -> sp.csr_matrix:
        """Generate sparse matrix for artists' tags, shape [max_artist_id + 1, max_tag_id + 1]
        :param artist_tagged_record: The total tagged record.
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
    def generate_user_tag_dependence_score(artist_tagged_record: pd.DataFrame) -> Dict[int, sp.csr_matrix]:
        """Cal each user's dependence on each tag"""

        # user-tag-tf
        user_tag_summary = artist_tagged_record.groupby(["userID", "tagID"], as_index=False)\
            .agg({"artistID": "count"})\
            .rename(columns={"artistID": "tag_num"})
        user_tag_summary["total_tag_num"] = user_tag_summary.groupby("userID")["tag_num"].transform("sum")
        user_tag_summary["tag_tf"] = user_tag_summary["tag_num"] / user_tag_summary["total_tag_num"]

        # user-tag-idf
        n = artist_tagged_record["userID"].nunique()  # total tagging users
        tag_user_summary = artist_tagged_record.groupby("tagID", as_index=False)\
            .agg({"userID": "nunique"})\
            .rename(columns={"userID": "tag_num"})
        tag_user_summary["tag_idf"] = np.log(n / (tag_user_summary["tag_num"] + 1))










