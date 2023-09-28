from __future__ import absolute_import

import tensorflow.keras
import tensorflow as tf
import numpy as np


def map_weight(inputs):
    x_in = inputs[0]
    y_in = inputs[1]
    a = inputs[2]
    b = inputs[3]
    return tf.where(x_in < y_in, x=a, y=b)


class GraphAttention(tensorflow.keras.layers.Layer):
    def __init__(self,
                 F_,
                 attn_heads=1,  # 多头注意力
                 attn_heads_reduction='concat',  # {'concat', 'average'}  # 注意力特征组合方式
                 dropout_rate=0.6,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='he_normal',
                 bias_initializer='he_normal',
                 attn_kernel_initializer='he_normal',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = tensorflow.keras.activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = tensorflow.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tensorflow.keras.initializers.get(bias_initializer)
        self.attn_kernel_initializer = tensorflow.keras.initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = tensorflow.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tensorflow.keras.regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = tensorflow.keras.regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = tensorflow.keras.regularizers.get(activity_regularizer)

        self.attn_kernel_constraint = tensorflow.keras.constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.filters_weights = []       # 残差降维滤波
        self.attn_kernels = []  # Attention kernels for attention heads

        # 采用多头注意力，concat表示特征拼接，average表示特征求平均
        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]  # 输入特征维度
        decision_F = input_shape[0][1]
        # Initialize weights for each attention head
        self.alpha = self.add_weight(shape=(decision_F//2, decision_F//2),  # W：N*N
                               initializer=self.kernel_initializer,
                               regularizer=self.kernel_regularizer,
                               trainable=True,
                               name='alph')

        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),  # W：F*F_
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     trainable=True,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       trainable=True,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               trainable=True,

                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 trainable=True,

                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])

            # filters_weight
            filters_weight = self.add_weight(shape=(1, F, self.F_),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             trainable=True,

                                             name='kernel_{}'.format(head))
            self.filters_weights.append(filters_weight)

        self.built = True
        super(GraphAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        X = inputs[0]  # Node features (b x N x F)
        A = inputs[1]  # Adjacency matrix (b x N x N)
        outputs = []
        self.dense_view = []
        self.dense_output = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')

            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            self.features = tensorflow.keras.backend.dot(X, kernel)  # (b x N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            self.attn_for_self_1 = tensorflow.keras.backend.dot(self.features, attention_kernel[0])    # (b x N x 1), [a_1]^T [Wh_i]
            self.attn_for_neighs_1 = tensorflow.keras.backend.dot(self.features, attention_kernel[1])  # (b x N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            self.dense_1 = self.attn_for_self_1 + tf.transpose(self.attn_for_neighs_1, perm=[0, 2, 1])  # (b x N x N) via broadcasting

            # Add nonlinearty
            self.dense_2 = tensorflow.keras.layers.LeakyReLU(alpha=0.2)(self.dense_1)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - A)  # A :b x N x N
            self.dense_2 += mask

            # Apply softmax to get attention coefficients
            self.dense_3 = tensorflow.keras.backend.softmax(self.dense_2)  # (b x N x N)

            # Apply dropout to features and attention coefficients
            dropout_attn = tensorflow.keras.layers.Dropout(self.dropout_rate)(self.dense_3)  # (b x N x N)
            dropout_feat = tensorflow.keras.layers.Dropout(self.dropout_rate)(self.features)  # (b x N x F')

            # Linear combination with neighbors' features
            node_features = tensorflow.keras.backend.batch_dot(dropout_attn, dropout_feat)  # (b x N x F')

            if self.use_bias:
                node_features = tensorflow.keras.backend.bias_add(node_features, self.biases[head])
            '''
            # residual connection
            num_1 = X.get_shape()[-1]
            num_2 = node_features.get_shape()[-1]
            if num_1 != num_2:
                short_cut = tf.nn.conv1d(X, self.filters_weights[head], 1, padding='SAME')
            else:
                short_cut = X
            
            node_features = node_features + short_cut
            '''
            outputs.append(node_features)  # k x b x N x F'
            self.dense_view.append(self.dense_3)  # k x b x N x N

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = tensorflow.keras.backend.concatenate(outputs)  # (N x KF')
        else:
            output = tensorflow.keras.backend.mean(tensorflow.keras.backend.stack(outputs), axis=0)  # N x F'

        output = self.activation(output)

        self.dense_view_1 = tensorflow.keras.backend.mean(tensorflow.keras.backend.stack(self.dense_view), axis=0)
        self.dense_view_2 = tf.expand_dims(self.dense_view_1, -1)
        self.dense_view_2 = tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(self.dense_view_2)
        self.dense_view_2 = tensorflow.keras.backend.squeeze(self.dense_view_2, -1)
        self.dense_temp_1 = tensorflow.keras.backend.dot(self.dense_view_2, self.alpha)
        self.dense_output_1 = tensorflow.keras.backend.sum(self.dense_temp_1, axis=-1)
        self.dense_output_2 = tensorflow.keras.backend.sum(self.dense_output_1, axis=-1)
        self.dense_output_3 = tensorflow.keras.backend.reshape(self.dense_output_2, (-1, 1))
        self.dense_output_4 = tensorflow.keras.layers.Activation(activation='sigmoid')(self.dense_output_3)
        one = tf.ones_like(self.dense_view_1)
        zero = tf.zeros_like(self.dense_view_1)

        self_loop = tensorflow.keras.backend.constant(np.identity(256))

        self.dense_4 = tf.map_fn(fn=map_weight, elems=(self.dense_view_1 + self_loop, self.dense_output_4, one, zero), dtype=tf.float32)

        return output, self.dense_4

    def compute_output_shape(self, input_shape):
        output_shape_1 = input_shape[0][0], self.output_dim
        return output_shape_1

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {'F_': self.F_,
                  'attn_heads': self.attn_heads,
                  'attn_heads_reduction': self.attn_heads_reduction,
                  'dropout_rate': self.dropout_rate,
                  'activation': self.activation,
                  'use_bias': self.use_bias,
                  'kernel_initializer': self.kernel_initializer,
                  'bias_initializer': self.bias_initializer,
                  'attn_kernel_initializer': self.attn_kernel_initializer,
                  'kernel_regularizer': self.kernel_regularizer,
                  'bias_regularizer': self.bias_regularizer,
                  'attn_kernel_regularizer': self.attn_kernel_regularizer,
                  'activity_regularizer': self.activity_regularizer,
                  'attn_kernel_constraint': self.attn_kernel_constraint,
                  'supports_masking': self.supports_masking,
                  'kernels': self.kernels,
                  'biases': self.biases,
                  'filters_weights': self.filters_weights,
                  'attn_kernels': self.attn_kernels,
                  'dense_1': self.dense_1,
                  'dense_2': self.dense_2,
                  'dense_3': self.dense_3,
                  'dense_view': self.dense_view,
                  'dense_output_4': self.dense_output_4,
                  'attn_for_self_1': self.attn_for_self_1,
                  'attn_for_neighs_1': self.attn_for_neighs_1,
                  'features': self.features
                  }
        base_config = super(GraphAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

