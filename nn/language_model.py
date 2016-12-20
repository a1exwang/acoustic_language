from keras.layers import Dense, Conv1D, LSTM, Conv2D, GRU
from keras.layers import Activation, AveragePooling2D, Dropout, Flatten, Reshape, Layer
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD
from nn.helpers import nhot_acc, mynhot_acc
from data.maps_constants import *
import numpy as np


class Model:
    def __init__(self, timestamps):
        # input_shape = (batch_size, timestamps, 1, input_freq_width, 1)
        self.input_freq_width = 88*3
        self.timestamps = timestamps

        model = Sequential()
        model.add(TimeDistributed(Conv2D(nb_filter=16,
                                         nb_row=13*3+1,
                                         nb_col=1,
                                         border_mode='valid'),
                                  input_shape=(self.timestamps, 1, self.input_freq_width, 1)))
        model.add((Activation("relu")))
        model.add(TimeDistributed(AveragePooling2D(pool_size=(3, 1))))

        model.add(TimeDistributed(Conv2D(nb_filter=10,
                                         nb_row=16,
                                         nb_col=1,
                                         border_mode='valid')))
        model.add(Activation("relu"))
        model.add(TimeDistributed(AveragePooling2D(pool_size=(5, 1))))

        model.add(TimeDistributed(Flatten()))

        model.add(TimeDistributed(Dense(output_dim=256)))
        model.add(Activation("relu"))

        # model.add(Bidirectional(LSTM(output_dim=88, return_sequences=False)))
        #
        # model.add(Reshape(target_shape=(self.timestamps, self.input_freq_width),
        #                   input_shape=(self.timestamps, 1, self.input_freq_width, 1)))
        model.add(LSTM(output_dim=88, return_sequences=False))

        # model.add(Dropout(0.2))
        model.add(Dense(output_dim=SEMITONES_ON_PIANO))
        model.add(Activation("softmax"))
        # model.add(Reshape((SEMITONES_ON_PIANO * timestamps,)))

        model.summary()

        opt = RMSprop(lr=1e-5)
        # opt = SGD(lr=0.00001, momentum=0.9, decay=0.0005)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=[nhot_acc, mynhot_acc])
        self.model = model

    def get_model(self):
        return self.model

    def train(self, x_batch, y_batch):
        return self.model.train_on_batch(
            x_batch,
            y_batch)

    def make_input(self, x, y, batch_size, timestamps):
        assert(x.shape[0] == y.shape[0])
        big_batch_size = batch_size * timestamps
        data_count = x.shape[1] // big_batch_size

        for i in range(data_count):
            x_batch = x[i*big_batch_size:(i+1)*big_batch_size, :]
            y_batch = y[i*big_batch_size:(i+1)*big_batch_size, :]
            y_batch_seq = np.reshape(y[i*big_batch_size:(i+1)*big_batch_size:timestamps, :],
                                     [batch_size, SEMITONES_ON_PIANO])
            x_batch = np.reshape(x_batch, [batch_size, timestamps, 1, self.input_freq_width, 1])
            y_batch = np.reshape(y_batch, [batch_size, timestamps * SEMITONES_ON_PIANO])

            # yield (x_batch, y_batch)
            yield (x_batch, y_batch_seq)

    def save_to_file(self, file_path):
        pass

    def load_from_file(self, file_path):
        pass

