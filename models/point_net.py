import tensorflow.keras
from tensorflow.keras import layers
from transform_XYZRGB_layer import transform_XYZRGB


class PGNet(layers.Layer):
    def __init__(self, F, N, filter_num):
        super(PGNet, self).__init__()
        self.F = F
        self.N = N
        self.filter_num = filter_num
        # point_net
        self.p_cov1 = tensorflow.keras.layers.Conv2D(self.filter_num, [1, self.F], kernel_initializer='he_normal', padding='VALID', strides=(1, 1))  # F为输入特征维度
        self.act_1 = tensorflow.keras.layers.LeakyReLU(alpha=0.2)
        self.bn_1 = layers.BatchNormalization()
        self.p_cov2 = tensorflow.keras.layers.Conv2D(self.filter_num*2, [1, 1], kernel_initializer='he_normal', padding='VALID', strides=[1, 1])
        self.act_2 = tensorflow.keras.layers.LeakyReLU(alpha=0.2)
        self.bn_2 = layers.BatchNormalization()
        self.p_cov3 = tensorflow.keras.layers.Conv2D(self.filter_num*4, [1, 1], kernel_initializer='he_normal', padding='VALID', strides=[1, 1])
        self.act_3 = tensorflow.keras.layers.LeakyReLU(alpha=0.2)
        self.bn_3 = layers.BatchNormalization()
        self.p_maxpooling = tensorflow.keras.layers.MaxPool2D([self.N, 1], padding='VALID')  # 节点数
        self.flatten = tensorflow.keras.layers.Flatten()
        self.p_fc1 = tensorflow.keras.layers.Dense(self.filter_num*4, kernel_initializer='he_normal')
        self.act_4 = tensorflow.keras.layers.LeakyReLU(alpha=0.2)
        self.bn_4 = layers.BatchNormalization()
        self.p_fc2 = tensorflow.keras.layers.Dense(self.filter_num*2, kernel_initializer='he_normal')
        self.act_5 = tensorflow.keras.layers.LeakyReLU(alpha=0.2)
        self.bn_5 = layers.BatchNormalization()
        self.transform = transform_XYZRGB(self.F, self.filter_num*2)
        self.p_cov4 = tensorflow.keras.layers.Conv2D(self.filter_num, [1, self.F], kernel_initializer='he_normal', padding='VALID', strides=[1, 1])
        self.act_6 = tensorflow.keras.layers.LeakyReLU(alpha=0.2)
        self.bn_6 = layers.BatchNormalization()
        self.p_cov5 = tensorflow.keras.layers.Conv2D(self.filter_num, [1, 1], kernel_initializer='he_normal', padding='VALID', strides=[1, 1])
        self.act_7 = tensorflow.keras.layers.LeakyReLU(alpha=0.2)
        self.bn_7 = layers.BatchNormalization()

    def call(self, inputs, training=None):
        X = inputs
        if len(X.get_shape()) < 4:
            X = tensorflow.keras.backend.expand_dims(X, -1)

        out = self.p_cov1(X)
        out = self.act_1(out)
        out = self.bn_1(out, training=training)
        out = self.p_cov2(out)
        out = self.act_2(out)
        out = self.bn_2(out, training=training)
        out = self.p_cov3(out)
        out = self.act_3(out)
        out = self.bn_3(out, training=training)
        out = self.p_maxpooling(out)
        out = self.flatten(out)
        out = self.p_fc1(out)
        out = self.act_4(out)
        out = self.bn_4(out)
        out = self.p_fc2(out)
        out = self.act_5(out)
        out = self.bn_5(out)
        p_X_transform = tensorflow.keras.backend.squeeze(X, -1)
        p_transform = self.transform(out)
        p_point_cloud_transformed = tensorflow.keras.backend.batch_dot(p_X_transform, p_transform)
        out = tensorflow.keras.backend.expand_dims(p_point_cloud_transformed, -1)
        out = self.p_cov4(out)
        out = self.act_6(out)
        out = self.bn_6(out)
        out = self.p_cov5(out)
        out = self.act_7(out)
        out = self.bn_7(out)
        pointnet_output = tensorflow.keras.backend.squeeze(out, -2)
        return pointnet_output

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {'F': self.F, 'N': self.N, 'filter_num': self.filter_num}
        base_config = super(PGNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))