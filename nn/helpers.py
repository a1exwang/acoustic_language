import keras.metrics
import keras.backend as K
from data.maps_constants import SEMITONES_ON_PIANO


def nhot_acc(y_true, y_pred):
    # n_activated = K.sum(y_true, axis=1)
    return keras.metrics.top_k_categorical_accuracy(y_true=y_true, y_pred=y_pred)


def mynhot_acc(y_true, y_pred, threshold=0.1):
    zero_or_one = (K.sign(y_pred - threshold) / 2 + 0.5)
    return 1 - K.sum(K.abs(y_true - zero_or_one)) / SEMITONES_ON_PIANO
