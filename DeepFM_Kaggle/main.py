
"""
The application of DeepFM on `Porto Seguro's Safe Driver Prediction(Kaggle Match)`
"""

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from DeepFM_Kaggle.model_utils import DeepFM
from DeepFM_Kaggle.data_utils import DataGenerator, TRAIN, TEST
from constant import PROJECT_HOME
from _utils.utensorflow import get_session_config


package_home = os.path.join(PROJECT_HOME, "DeepFM")
BATCH_SIZE = 64
EPOCHS = 5
MODEL_DIR = os.path.join(package_home, "model")


def get_callbacks():
    checkpoint_file = os.path.join(MODEL_DIR, "model_{epoch:02d}_{val_loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath=checkpoint_file, monitor="val_loss", save_weights_only=False,
                                 verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    return [checkpoint, early_stop]


if __name__ == '__main__':
    tf.reset_default_graph()
    K.clear_session()
    K.set_session(tf.Session(config=get_session_config()))
    callbacks = get_callbacks()

    model = DeepFM()