from keras.layers import Dense, Activation
from keras.models import Sequential
from data.maps import Maps, RedisCache
from data.cqt_frame_transformer import Transformer
import nn.mlp
import nn.cnn
import numpy as np

# my_model = nn.mlp.Model()
my_model = nn.cnn.Model()
model = my_model.get_model()

display_freq = 200
test_freq = 1000

transformer = Transformer()
db = Maps('../db', RedisCache('audio'), transformer=transformer)
db.warm_up()
iter_count = 0
mean_loss_list = []
acc_list = []
batch_size = 32
epoch = 5
for e in range(epoch):
    file_count = 0
    for (file_path, (cqt, midis)) in db.pieces(shuffle=False):
        file_count += 1
        total_files = db.total_files()
        # print("Epoch %d, file(%d/%d) %s" % (e, file_count, total_files, file_path))
        for (x_batch, y_batch) in db.make_batch(cqt, midis, batch_size):
            iter_count += 1
            loss, acc = model.train_on_batch(
                np.reshape(x_batch, [x_batch.shape[0], x_batch.shape[1], 1]),
                y_batch)

            mean_loss_list.append(loss)
            acc_list.append(acc)

            if iter_count % display_freq == 0:
                print("Iter %d, mean loss %0.5f, batch loss %0.5f, acc %f" %
                      (iter_count,
                       np.mean(mean_loss_list),
                       loss,
                       np.mean(acc_list)))
                mean_loss_list = []
                acc_list = []
