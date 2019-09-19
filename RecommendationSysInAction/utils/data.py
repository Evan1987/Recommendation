"""Common methods to load data"""

import os
import pandas as pd
from typing import Dict, Union


_PATH = r"F:\for learn\Python\推荐系统开发实战\章节代码\data"


def load_data(file: str, sep: str = "\t", dtype: Union[Dict, type] = None):
    contents = []
    with open(file, "r", encoding="utf-8") as f:
        columns = next(f).strip().split(sep)
        for line in f:
            line = line.strip()
            if line:
                contents.append(line.split(sep))

    data = pd.DataFrame(contents, columns=columns)
    if dtype is None:
        return data
    if isinstance(dtype, dict):
        for key in dtype.keys():
            if key not in columns:
                raise ValueError(f"Unknown `{key}` in columns, given {columns}.")
        return data.astype(dtype)
    if isinstance(dtype, type):
        return data.astype(dtype)


class LastFM(object):
    _dir = os.path.join(_PATH, "lastfm-2k")
    _sep = "\t"
    n_users = 1892
    n_artists = 17632
    n_tags = 11946

    max_artist_id = 18745
    max_tag_id = 12648

    @classmethod
    def load_artists_data(cls) -> pd.DataFrame:
        # columns: [id,name,url,pictureURL]
        dtype = {"id": int}
        return load_data(os.path.join(cls._dir, "artists.dat"), sep=cls._sep, dtype=dtype)

    @classmethod
    def load_tags_data(cls) -> pd.DataFrame:
        # columns: [tagID,tagValue]
        dtype = {"tagID": int}
        return load_data(os.path.join(cls._dir, "tags.dat"), sep=cls._sep, dtype=dtype)

    @classmethod
    def load_rates_data(cls) -> pd.DataFrame:
        # columns: [userID,artistID,weight]
        return load_data(os.path.join(cls._dir, "user_artists.dat"), sep=cls._sep, dtype=int)

    @classmethod
    def load_user_tagged_artists_data(cls) -> pd.DataFrame:
        # columns: [userID,artistID,tagID,day,month,year]
        return load_data(os.path.join(cls._dir, "user_taggedartists.dat"), sep=cls._sep, dtype=int)


class HotelMess:

    n_pois = 650

    @classmethod
    def load_data(cls):
        return pd.read_csv(os.path.join(_PATH, "hotel-mess/hotel-mess.csv"), engine="python", encoding="GBK")
