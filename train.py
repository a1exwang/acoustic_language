from data.maps import Maps, RedisCache
from data.cqt_frame_transformer import Transformer
import nn.cnn
import nn.language_model
import numpy as np

# model = nn.mlp.Model()
# model = nn.cnn.Model()
model = nn.language_model.Model(timestamps=10)


display_freq = 20
test_freq = 1000
batch_size = 4
epoch = 5

transformer = Transformer()
db = Maps('../db', RedisCache('audio'), transformer=transformer)
db.warm_up()

iter_count = 0
mean_loss_list = []
acc_list = []
nacc_list = []
for e in range(epoch):
    file_count = 0
    for (file_path, (cqt, midis)) in db.pieces(shuffle=False):
        file_count += 1
        total_files = db.total_files()
        # print("Epoch %d, file(%d/%d) %s" % (e, file_count, total_files, file_path))
        # for (x_batch, y_batch) in model.make_input(cqt, midis, batch_size=batch_size):
        for (x_batch, y_batch) in model.make_input(cqt, midis, batch_size=batch_size, timestamps=10):
            iter_count += 1

            loss, acc, nacc = model.train(x_batch, y_batch)
            mean_loss_list.append(loss)
            acc_list.append(acc)
            nacc_list.append(nacc)

            if iter_count % display_freq == 0:
                print("Iter %d, mean loss %0.5f, batch loss %0.5f, acc %f, nacc %f" %
                      (iter_count,
                       np.mean(mean_loss_list),
                       loss,
                       np.mean(acc_list),
                       np.mean(nacc_list)))
                mean_loss_list = []
                acc_list = []
