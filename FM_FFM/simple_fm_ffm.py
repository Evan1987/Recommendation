"""
Model: FM & FFM
Dataset: MovieLens  # user -> [1, 943], item -> [1, 1682]
Framework: xlearn
"""

import os
import numpy as np
import xlearn as xl
from FM_FFM.data_utils import data_path, TEST_Y_TRUE, package_home
from sklearn.metrics import mean_squared_error, mean_absolute_error


PARAM = {"epoch": 10, "task": "reg", "lr": 0.2, "lambda": 2e-4,
         "metric": "mae", "log": "F:/test/", "seed": 0, "k": 4}
MODEL_DIR = os.path.join(package_home, "model")
TEMP_PREDICTION_FILE = os.path.join(data_path, "temp_pred.txt")


def read_xlearn_pred(file: str):
    with open(file, "r") as f:
        result = f.readlines()
    return np.asarray([r.strip() for r in result]).astype(float)


def train_model(model_type: str):
    if model_type == "fm":
        model = xl.create_fm()
        train_data = os.path.join(data_path, "train.svm")
        test_data = os.path.join(data_path, "test.svm")
    elif model_type == "ffm":
        model = xl.create_ffm()
        train_data = os.path.join(data_path, "train.ffm")
        test_data = os.path.join(data_path, "test.ffm")
    else:
        raise ValueError(f"Unknown model type {model_type}.")

    output_model = os.path.join(MODEL_DIR, f"{model_type}.model")
    model.setTrain(train_data)
    model.setValidate(test_data)
    model.setNoBin()
    model.disableLockFree()
    model.fit(PARAM, model_path=output_model)
    model.setTest(test_data)
    model.predict(output_model, out_path=TEMP_PREDICTION_FILE)

    y_pred = read_xlearn_pred(TEMP_PREDICTION_FILE)
    mse = mean_squared_error(TEST_Y_TRUE, y_pred)
    mae = mean_absolute_error(TEST_Y_TRUE, y_pred)
    print(f"Model {model_type}  mse: {mse}, mae: {mae}")


if __name__ == '__main__':
    train_model("fm")  # mse: 0.8824899110219306, mae: 0.7388163868504772
    train_model("ffm")  # mse: 0.8716266362343029, mae: 0.7352675924708377
