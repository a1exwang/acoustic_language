from keras.layers import Dense, Activation, Conv1D, AveragePooling1D, Dropout, Flatten
from keras.models import Sequential
import keras.backend as K
import keras.metrics


def nhot_acc(y_true, y_pred):
    # n_activated = K.sum(y_true, axis=1)
    return keras.metrics.top_k_categorical_accuracy(y_true=y_true, y_pred=y_pred)


class Model:
    def __init__(self):
        model = Sequential()

        model.add(Conv1D(nb_filter=16,
                         filter_length=13*3+1,
                         border_mode='valid',
                         input_shape=(88*3, 1)))
        model.add(Activation("relu"))
        model.add(AveragePooling1D(3))

        model.add(Conv1D(nb_filter=10, filter_length=16))
        model.add(Activation("relu"))
        model.add(AveragePooling1D(pool_length=2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(output_dim=88))
        model.add(Activation("softmax"))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer='adagrad',
                      metrics=[nhot_acc])
        self.model = model

    def get_model(self):
        return self.model

    def save_to_file(self, file_path):
        pass

    def load_from_file(self, file_path):
        pass
