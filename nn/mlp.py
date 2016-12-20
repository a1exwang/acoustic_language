from keras.layers import Dense, Activation
from keras.models import Sequential
from nn.helpers import nhot_acc


class Model:
    def __init__(self):
        model = Sequential()

        model.add(Dense(output_dim=256, input_dim=88*3))
        model.add(Activation("sigmoid"))
        model.add(Dense(output_dim=88))
        model.add(Activation("softmax"))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adagrad',
                      metrics=[nhot_acc])
        self.model = model

    def get_model(self):
        return self.model

    def train(self, x_batch, y_batch):
        return self.model.train_on_batch(x_batch, y_batch)

    def save_to_file(self, file_path):
        pass

    def load_from_file(self, file_path):
        pass
