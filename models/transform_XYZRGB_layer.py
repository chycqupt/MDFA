import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.constraints import Constraint
import tensorflow.keras.backend as K


class Between(Constraint):
    def __init__(self, min_value=-0.1, max_value=0.1):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

class transform_XYZRGB(tensorflow.keras.layers.Layer):
    def __init__(self,
                 k,
                 m):
        super().__init__()  # 必须写
        self.k = k
        self.m = m

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weights",
            shape=[self.m, self.k*self.k],
            initializer=tensorflow.keras.initializers.get('he_normal'),
            trainable=True,

            dtype=tf.float32
        )
        self.b = self.add_weight(
            name="biases",
            shape=[self.k*self.k],
            initializer=tensorflow.keras.initializers.get('zeros'),
            trainable=True,

            dtype=tf.float32
        )  # b一般是全0

    def call(self, input):
        transform_1 = tensorflow.keras.backend.dot(input, self.w)
        self.b.assign_add(tensorflow.keras.backend.flatten(tf.convert_to_tensor(tf.eye(self.k))))
        transform_2 = tensorflow.keras.backend.bias_add(transform_1, self.b)

        return tensorflow.keras.backend.reshape(transform_2, [-1, self.k, self.k])