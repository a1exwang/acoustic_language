from keras.layers import Dense, Conv1D, LSTM, TimeDistributed
from keras.layers import Activation, AveragePooling1D, Dropout, Flatten, Reshape, Layer
from keras.models import Sequential
from nn.helpers import nhot_acc
import numpy as np


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

    def train(self, x_batch, y_batch):
        return self.model.train_on_batch(
            np.reshape(x_batch, [x_batch.shape[0], x_batch.shape[1], 1]),
            y_batch)

    def make_input(self, x, y, batch_size):
        assert(x.shape[0] == y.shape[0])
        data_count = x.shape[1] // batch_size

        for i in range(data_count):
            x_batch = x[i*batch_size:(i+1)*batch_size, :]
            y_batch = y[i*batch_size:(i+1)*batch_size, :]

            yield (x_batch, y_batch)

    def save_to_file(self, file_path):
        pass

    def load_from_file(self, file_path):
        pass
