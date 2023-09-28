import tensorflow.keras
import tensorflow as tf


def map_weight(inputs):
    x_in = inputs[0]
    y_in = inputs[1]
    a = inputs[2]
    b = inputs[3]
    return tf.where(x_in < y_in, x=a, y=b)


class AD_layer(tensorflow.keras.layers.Layer):
    def __init__(self):
        super(AD_layer, self).__init__()

    def call(self, inputs):
        self.adj = tf.map_fn(fn=map_weight, elems=inputs, dtype=tf.float32)
        return self.adj

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {'adj': self.adj}
        base_config = super(AD_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))