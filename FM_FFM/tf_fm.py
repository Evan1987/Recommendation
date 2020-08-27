
"""
Model: FM & FFM
Dataset: MovieLens  # user -> [1, 943], item -> [1, 1682]
Framework: tensorflow
"""

import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from evan_utils.utensorflow import get_session_config
from FM_FFM.data_utils import package_home, DATA_ALIAS, TFFMDataSet, TEST_Y_TRUE
from FM_FFM.model_utils import FM


EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.2

MODEL_DIR = os.path.join(package_home, "model")


def get_callbacks():
    checkpoint_file = os.path.join(MODEL_DIR, "model_{epoch:02d}_{val_loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath=checkpoint_file, monitor="val_loss", save_weights_only=False,
                                 verbose=1, save_best_only=True)
    return [checkpoint]


train_data = TFFMDataSet(DATA_ALIAS["train"], BATCH_SIZE)
test_data = TFFMDataSet(DATA_ALIAS["test"], BATCH_SIZE)


if __name__ == '__main__':
    tf.reset_default_graph()
    K.clear_session()
    K.set_session(tf.Session(config=get_session_config()))
    callbacks = get_callbacks()

    fm = FM(4, LEARNING_RATE)
    fm.summary()
    fm.plot_model(os.path.join(MODEL_DIR, "fm.png"))

    saved_model_file = os.path.join(MODEL_DIR, "fm.h5")

    if os.path.exists(saved_model_file):
        fm.load_model(saved_model_file)
    else:
        fm.train(train_data, test_data, EPOCHS, callbacks)
        fm.save_model(os.path.join(MODEL_DIR, "fm.h5"))

    y_pred = fm.predict(DATA_ALIAS["test"]["user"].values, DATA_ALIAS["test"]["item"].values)
    mse = mean_squared_error(TEST_Y_TRUE, y_pred)
    mae = mean_absolute_error(TEST_Y_TRUE, y_pred)

    print(f"FM tf Model mse: {mse}, mae: {mae}")  # mse: 1.5467057672010385, mae: 1.0149061494326161
