
"""
Manipulate data using given raw data and transformers
"""
import os
import logging
from FM_FFM.data_utils import FMDataTransformer, FFMDataTransformer, DATA_ALIAS, data_path
from sklearn.datasets import dump_svmlight_file


logging.basicConfig(format='%(asctime)s [line: %(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger("logger")


def make_fm_data():
    fm_data_transformer = FMDataTransformer()
    fm_data_transformer.fit(DATA_ALIAS["train"])

    for key in ["train", "test"]:
        X, y = fm_data_transformer.transform(DATA_ALIAS[key])
        dump_svmlight_file(X, y, os.path.join(data_path, f"{key}.svm"))
        LOGGER.info(f"Save {key} svm data successfully!")


def make_ffm_data():
    ffm_data_transformer = FFMDataTransformer()
    ffm_data_transformer.fit(DATA_ALIAS["train"])

    for key in ["train", "test"]:
        X = ffm_data_transformer.transform(DATA_ALIAS[key])
        X.to_csv(os.path.join(data_path, f"{key}.ffm"), index=False, header=False)
        LOGGER.info(f"Save {key} ffm data successfully!")


if __name__ == '__main__':
    make_fm_data()
    make_ffm_data()
