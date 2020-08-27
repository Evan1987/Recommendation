
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from NCF.model_utils import NCF
from NCF.data_utils import DataGenerator, load_data
from NCF.callbacks import MetricEvaluation
from evan_utils.utensorflow import get_session_config
from constant import PROJECT_HOME


package_home = os.path.join(PROJECT_HOME, "NCF")
MODEL_DIR = os.path.join(package_home, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 20
TOP_K = 5
SEED = 0
EMBEDDING_SIZE = 16


def get_callbacks(val_data: pd.DataFrame):
    checkpoint_file = os.path.join(MODEL_DIR, "model_{epoch:02d}_{val_loss:.4f}.hdf5")
    checkpoint = ModelCheckpoint(filepath=checkpoint_file, monitor="val_loss", save_weights_only=False,
                                 verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    metric_monitor = MetricEvaluation(val_data, verbose=True, topK=TOP_K, predict_batch_size=128)
    return [checkpoint, metric_monitor, early_stopping]


if __name__ == '__main__':
    tf.reset_default_graph()
    K.clear_session()
    K.set_session(tf.Session(config=get_session_config()))

    model = NCF(k=EMBEDDING_SIZE, learning_rate=1e-3, dropout=0)
    model.summary()
    model.plot_model(os.path.join(MODEL_DIR, "ncf.png"))
    saved_model_file = os.path.join(MODEL_DIR, "ncf.h5")
    if os.path.exists(saved_model_file):
        model.load_model(saved_model_file)
    else:
        train, val = load_data()
        train_data = DataGenerator(train, batch_size=BATCH_SIZE, seed=SEED, is_train=True)
        eval_data = DataGenerator(val, batch_size=BATCH_SIZE, seed=SEED, is_train=False)
        callbacks = get_callbacks(val)
        model.train(train_data, eval_data, epochs=EPOCHS, callbacks=callbacks)
        model.save_model(saved_model_file)
