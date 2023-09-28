from __future__ import division
import tensorflow as tf
import tensorflow.keras
from graph_attention_layer import GraphAttention
from AD_layer import AD_layer
from tensorflow.keras.regularizers import l2
from vgg_face_model import VGG16
from point_net import PGNet
import os

class MDFA(tf.keras.Model):
    def __init__(self, N, F, F_=8, n_attn_heads=8, dropout_rate=0.6, l2_reg= 5e-4 / 2):
        super(MDFA).__init__()
        self.F_ = F_
        self.n_attn_heads = n_attn_heads
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.N = N  # Number of nodes in the graph
        self.F = F  # Number of features
        self.N_ = N/2

    def AD_block(self, adj_1):
        adj_1_temp_1 = tf.expand_dims(adj_1, -1)
        adj_1_temp_2 = tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(adj_1_temp_1)
        adj_1_temp_3 = tensorflow.keras.layers.Flatten()(adj_1_temp_2)
        k_1 = tensorflow.keras.layers.Dense(1, kernel_initializer='he_normal', activation='sigmoid')(adj_1_temp_3)
        one = tf.ones_like(adj_1)
        zero = tf.zeros_like(adj_1)
        adj_1_1 = AD_layer()([adj_1, k_1, one, zero])
        return adj_1_1

    def VGG(self, uv_tex_in):
        base_model = VGG16(weights='vggface', input_tensor=uv_tex_in)
        uv_tex_f1 = base_model.get_layer('fc6').output

        for layer in base_model.layers:
            layer.trainable = True
        return uv_tex_f1

    def call(self, inputs):
        X_in = inputs[0]
        A_in = inputs[1]
        uv_tex_in = inputs[2]

        # 1
        pn_1 = PGNet(self.F, self.N, filter_num=16)(X_in)
        gat_1, adj_1 = GraphAttention(self.F_, attn_heads=self.n_attn_heads,
                                      dropout_rate=self.dropout_rate,
                                      kernel_regularizer=l2(self.l2_reg),
                                      attn_heads_reduction='concat')([pn_1, A_in])
        adj_1 = self.AD_block(adj_1)
        pg_1 = tensorflow.keras.layers.BatchNormalization()(gat_1)

        X_in_transform = tensorflow.keras.layers.Conv1D(filters=64, kernel_size=1, padding='VALID', strides=1)(X_in)
        res_input_1 = tensorflow.keras.layers.add([pg_1, X_in_transform])

        # 2
        pn_2 = PGNet(self.F_ * self.n_attn_heads, self.N, filter_num=16)(res_input_1)
        gat_2, adj_2 = GraphAttention(self.F_, attn_heads=self.n_attn_heads,
                                      dropout_rate=self.dropout_rate,
                                      kernel_regularizer=l2(self.l2_reg),
                                      attn_heads_reduction='concat')([pn_2, adj_1])
        adj_2 = self.AD_block(adj_2)
        pg_2 = tensorflow.keras.layers.BatchNormalization()(gat_2)

        # 3
        pn_3 = PGNet(self.F_ * self.n_attn_heads, self.N, filter_num=32)(pg_2)
        gat_3, adj_3 = GraphAttention(self.F_, attn_heads=self.n_attn_heads,
                                      dropout_rate=self.dropout_rate,
                                      kernel_regularizer=l2(self.l2_reg),
                                      attn_heads_reduction='concat')([pn_3, adj_2])
        adj_3 = self.AD_block(adj_3)
        pg_3 = tensorflow.keras.layers.BatchNormalization()(gat_3)

        res_input_2 = tensorflow.keras.layers.add([pg_2, pg_3])

        # 4
        pn_4 = PGNet(self.F_ * self.n_attn_heads, self.N, filter_num=32)(res_input_2)
        gat_4, adj_4 = GraphAttention(self.F_, attn_heads=self.n_attn_heads,
                                      dropout_rate=self.dropout_rate,
                                      kernel_regularizer=l2(self.l2_reg),
                                      attn_heads_reduction='concat')([pn_4, adj_3])
        adj_4 = self.AD_block(adj_4)
        pg_4 = tensorflow.keras.layers.BatchNormalization()(gat_4)

        # 5
        pn_5 = PGNet(self.F_ * self.n_attn_heads, self.N, filter_num=64)(pg_4)
        gat_5, adj_5 = GraphAttention(self.F_, attn_heads=self.n_attn_heads,
                                      dropout_rate=self.dropout_rate,
                                      kernel_regularizer=l2(self.l2_reg),
                                      attn_heads_reduction='concat')([pn_5, adj_4])
        adj_5 = self.AD_block(adj_5)
        pg_5 = tensorflow.keras.layers.BatchNormalization()(gat_5)

        res_input_3 = tensorflow.keras.layers.add([pg_4, pg_5])

        # 6
        pn_6 = PGNet(self.F_ * self.n_attn_heads, self.N, filter_num=64)(res_input_3)
        gat_6, adj_6 = GraphAttention(self.F_, attn_heads=self.n_attn_heads,
                                      dropout_rate=self.dropout_rate,
                                      kernel_regularizer=l2(self.l2_reg),
                                      attn_heads_reduction='concat')([pn_6, adj_5])
        adj_6 = self.AD_block(adj_6)
        pg_6 = tensorflow.keras.layers.BatchNormalization()(gat_6)

        # 7
        pn_7 = PGNet(self.F_ * self.n_attn_heads, self.N, filter_num=64)(pg_6)
        gat_7, adj_7 = GraphAttention(128, attn_heads=self.n_attn_heads,
                                      dropout_rate=self.dropout_rate,
                                      kernel_regularizer=l2(self.l2_reg),
                                      attn_heads_reduction='average')([pn_7, adj_6])
        pg_7 = tensorflow.keras.layers.BatchNormalization()(gat_7)

        graph_attention_output = tf.expand_dims(pg_7, -1)
        graph_attention_output = tensorflow.keras.layers.MaxPool2D([self.N_/2, 1], padding='VALID', name='maxpool')(
            graph_attention_output)
        graph_attention_output = tensorflow.keras.layers.Flatten()(graph_attention_output)

        uv_tex_f1 = self.VGG(uv_tex_in)

        graph_attention_output = tensorflow.keras.layers.Dense(4096, kernel_initializer='he_normal', activation='relu')(graph_attention_output)
        graph_attention_output = tensorflow.keras.layers.BatchNormalization()(graph_attention_output)

        merge = tensorflow.keras.layers.concatenate([graph_attention_output, uv_tex_f1])
        D = tensorflow.keras.layers.Dense(4096, kernel_initializer='he_normal', activation='relu')(merge)
        D = tensorflow.keras.layers.BatchNormalization()(D)
        D = tensorflow.keras.layers.Dense(4096, kernel_initializer='he_normal', activation='relu')(D)
        D = tensorflow.keras.layers.BatchNormalization()(D)

        D = tensorflow.keras.layers.Dense(5, activation='relu')(D)
        output = tensorflow.keras.layers.Softmax(axis=-1)(D)
        return output