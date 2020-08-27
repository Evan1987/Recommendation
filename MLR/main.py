
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from MLR.data_utils import train_data, test_data, DataGenerator
from MLR.model_utils import MLR
from evan_utils.utensorflow import get_session_config
from constant import PROJECT_HOME


PACKAGE_HOME = os.path.join(PROJECT_HOME, "MLR")
BATCH_SIZE = 64
EPOCHS = 100
SEED = 0
D = train_data.shape[1] - 1  # The num of features
M = 4
MODEL_DIR = os.path.join(PACKAGE_HOME, "model")
saved_model_file = os.path.join(MODEL_DIR, "mlr.h5")
os.makedirs(MODEL_DIR, exist_ok=True)


def get_callbacks():
    checkpoint_file = os.path.join(MODEL_DIR, "model_{epoch:02d}_{val_loss:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath=checkpoint_file, monitor="val_loss", save_weights_only=False,
                                 verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    return [checkpoint, early_stop]


if __name__ == '__main__':
    tf.reset_default_graph()
    K.clear_session()
    K.set_session(tf.Session(config=get_session_config()))

    model = MLR(d=D, m=M, learning_rate=0.02)
    model.summary()
    model.plot_model(os.path.join(PACKAGE_HOME, "mlr.png"))

    test = DataGenerator(test_data, batch_size=1024)
    
    if os.path.exists(saved_model_file):
        model.load_model(saved_model_file)
    else:
        callbacks = get_callbacks()
        train_data, val_data = train_test_split(train_data, test_size=0.2)
        train = DataGenerator(train_data, batch_size=BATCH_SIZE, seed=SEED)
        val = DataGenerator(val_data, batch_size=BATCH_SIZE, seed=SEED)

        model.train(train, val, epochs=EPOCHS, callbacks=callbacks)
        model.save_model(saved_model_file)

    y_scores_total = []
    y_true_total = []
    for i in tqdm(range(len(test)), desc="Predict on test data"):
        x, y = test[i]
        y_scores = model.predict(x)
        y_scores_total.append(y_scores)
        y_true_total.append(y)

    y_scores_total = np.array(y_scores_total).flatten()
    y_true_total = np.array(y_true_total).flatten()
    y_pred_total = (y_scores_total >= 0.5).astype(int)

    auc = roc_auc_score(y_true_total, y_scores_total)
    print(f"The AUC: {auc:.4f}.")  # 0.9077
    print(classification_report(y_true_total, y_pred_total))
