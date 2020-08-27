
"""
The application of DeepFM on `Porto Seguro's Safe Driver Prediction(Kaggle Match)`
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from DeepFM_Kaggle.model_utils import DeepFM
from DeepFM_Kaggle.callbacks import GiniCheckPoint
from DeepFM_Kaggle.data_utils import DataGenerator, TRAIN, TEST, LABEL_COL, FEATURES
from constant import PROJECT_HOME
from evan_utils.utensorflow import get_session_config


package_home = os.path.join(PROJECT_HOME, "DeepFM_Kaggle")
BATCH_SIZE = 64
EPOCHS = 5
SEED = 0
MODEL_DIR = os.path.join(package_home, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def df_to_input(df: pd.DataFrame, batch_size: int = None):
    n = len(df)
    if batch_size is None or batch_size < 0:
        batch_size = n
    cur = 0
    output = []
    while cur < n:
        batch = df.iloc[cur: cur + batch_size]
        X = {feature: batch[feature].values.reshape(-1, 1) for feature in FEATURES}
        y = batch[LABEL_COL].values.reshape(-1, 1) if LABEL_COL in batch else None
        output.append((X, y))
        cur += batch_size
    return output


def get_callbacks(val_data: pd.DataFrame):
    X_val, y_val = df_to_input(val_data)[0]  # only one batch
    checkpoint_file = os.path.join(MODEL_DIR, "model_{epoch:02d}_{val_gini:.4f}.hdf5")
    gini_monitor = GiniCheckPoint(filepath=checkpoint_file, val_data=(X_val, y_val), verbose=True)
    early_stopping = EarlyStopping(monitor='val_gini', mode="max", patience=3)
    return [gini_monitor, early_stopping]


if __name__ == '__main__':
    tf.reset_default_graph()
    K.clear_session()
    K.set_session(tf.Session(config=get_session_config()))

    model = DeepFM(learning_rate=1e-3, k=8, model_type="afm", final_dnn=False)
    model.summary()
    model.plot_model(os.path.join(MODEL_DIR, f"{model.model_type}.png"))

    saved_model_file = os.path.join(MODEL_DIR, f"{model.model_type}.h5")
    if os.path.exists(saved_model_file):
        model.load_model(saved_model_file)
    else:
        train, val = train_test_split(TRAIN, test_size=0.2, random_state=SEED)
        train_data = DataGenerator(train, batch_size=BATCH_SIZE)
        eval_data = DataGenerator(val, batch_size=BATCH_SIZE)
        callbacks = get_callbacks(val)
        model.train(train_data, eval_data, EPOCHS, callbacks)
        model.save_model(saved_model_file)

    test_inputs = df_to_input(TEST, batch_size=1024)
    targets = [model.predict(X).reshape(-1) for X, _ in tqdm(test_inputs)]
    TEST[LABEL_COL] = np.hstack(targets)
    sub: pd.DataFrame = TEST[["id", LABEL_COL]]
    print(sub.head(10))
    sub.to_csv(os.path.join(package_home, f"{model.model_type}_submission.csv"), index=False)
    # deepfm: expect 0.25~0.26 in gini for 5 epoch
    # nfm:
