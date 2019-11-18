
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from _utils.utensorflow import get_session_config
from DeepFM.model_utils import DeepFM
from DeepFM.data_utils import DataGenerator, DATA_ALIAS, TEST_Y_TRUE, pad_genres
from constant import PROJECT_HOME


package_home = os.path.join(PROJECT_HOME, "DeepFM")
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-3
MODEL_DIR = os.path.join(package_home, "model")


def get_callbacks():
    checkpoint_file = os.path.join(MODEL_DIR, "model_{epoch:02d}_{val_loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath=checkpoint_file, monitor="val_loss", save_weights_only=False,
                                 verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    return [checkpoint, early_stop]


train, test = DATA_ALIAS["train"], DATA_ALIAS["test"]
train_data = DataGenerator(train, BATCH_SIZE)
test_data = DataGenerator(test, BATCH_SIZE)


if __name__ == '__main__':
    tf.reset_default_graph()
    K.clear_session()
    K.set_session(tf.Session(config=get_session_config()))
    callbacks = get_callbacks()

    dfm = DeepFM(k=20, learning_rate=LEARNING_RATE)
    dfm.summary()
    dfm.plot_model(os.path.join(MODEL_DIR, "fm.png"))

    saved_model_file = os.path.join(MODEL_DIR, "fm.h5")

    if os.path.exists(saved_model_file):
        dfm.load_model(saved_model_file)
    else:
        dfm.train(train_data, test_data, EPOCHS, callbacks)
        dfm.save_model(os.path.join(MODEL_DIR, "fm.h5"))

    y_pred = dfm.predict(
        users=test["user"].values,
        items=test["item"].values,
        occupations=test["occupation"].values,
        genres=pad_genres(test["genres"].values))
    mse = mean_squared_error(TEST_Y_TRUE, y_pred)
    mae = mean_absolute_error(TEST_Y_TRUE, y_pred)

    print(f"FM tf Model mse: {mse}, mae: {mae}")
