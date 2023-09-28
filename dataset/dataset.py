import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
import math


def rotate_point_cloud(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    batch_data_location = batch_data[:, :, 0:3]
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data_location[k, ...]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def shuffle_data(data, adj, uv_tex, labels):
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], adj[idx], uv_tex[idx], labels[idx]


class DataGenerator(Sequence):

    def __init__(self, train_x, train_uv_tex, train_y, batch_size=1, shuffle=False, train=True):
        self.batch_size = batch_size
        self.train_x = train_x
        self.train_uv_tex = train_uv_tex
        self.train_y = train_y
        self.indexes = np.arange(len(self.train_x))
        self.shuffle = shuffle
        self.on_epoch_end()
        self.train = train

    def __len__(self):
        return math.ceil(len(self.train_x) / float(self.batch_size))

    def __getitem__(self, index):
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_datas = [(self.train_x[k], self.train_uv_tex[k], self.train_y[k]) for k in batch_indexs]
        X, y = self._data_generation(batch_datas)

        return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _data_generation(self, batch_datas):
        feature = []
        uv_tex = []
        adj = []
        label = []

        for i in range(len(batch_datas)):
            train_feature, train_uv_tex, train_label = batch_datas[i]
            feature.append(train_feature)
            uv_tex.append(train_uv_tex)
            adj.append(np.ones((256, 256)))
            label.append(train_label)

        if self.train:
            feature, adj, uv_tex, label = rotate_point_cloud(jitter_point_cloud(np.array(feature))), np.array(adj), np.array(uv_tex), np.array(label)
        else:
            feature, adj, uv_tex, label = np.array(feature), np.array(adj), np.array(uv_tex), np.array(label)
        feature, adj, uv_tex, label = shuffle_data(feature, adj, uv_tex, label)
        F = [feature, adj, uv_tex]
        return F, label