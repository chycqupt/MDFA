from __future__ import division

import tensorflow as tf
import tensorflow.keras
from models import MDFA
from loss import LWD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from dataset.dataset import DataGenerator
import numpy as np
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train MDFA')
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--train_point_cloud_path", type=str, default="")
    parser.add_argument("--train_uv_tex_path", type=str, default="")
    parser.add_argument("--train_label_path", type=str, default="")
    parser.add_argument("--test_point_cloud_path", type=str, default="")
    parser.add_argument("--test_uv_tex_path", type=str, default="")
    parser.add_argument("--test_label_path", type=str, default="")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--F_", type=int, default=8)
    parser.add_argument("--n_attn_heads", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.6)
    parser.add_argument("--l2_reg", type=float, default=5e-4/2)
    parser.add_argument("--F_", type=int, default=8)
    parser.add_argument("--F_", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=5000)
    args = parser.parse_args()

    # load data
    feature_train = np.load(args.train_point_cloud_path)
    uv_tex_feature_train = np.load(args.train_uv_tex_path)
    distribution_label_train = np.load(args.train_label_path)

    feature_test = np.load(args.test_point_cloud_path)
    uv_tex_feature_test = np.load(args.test_uv_tex_path)
    distribution_label_test = np.load(args.test_label_path)

    N = feature_train.shape[1]  # Number of nodes in the graph
    F = feature_train.shape[2]  # Number of features

    train_model = MDFA(N, F)

    X_in = tensorflow.keras.layers.Input(shape=(N, F))
    A_in = tensorflow.keras.layers.Input(shape=(N, N))
    uv_tex_in = tensorflow.keras.layers.Input(shape=(224, 224, 3))

    out_put = train_model([X_in, A_in, uv_tex_in])

    # Build model
    model = Model(inputs=[X_in, A_in, uv_tex_in], outputs=out_put)

    model.summary()
    os.makedirs(args.save_path, exist_ok=True)
    optimizer = Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer,
                  loss=LWD,
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'), tf.keras.metrics.MeanAbsoluteError('mae')]
                  )

    checkpoint = ModelCheckpoint(args.save_path, monitor='val_rmse', verbose=1, save_best_only=True, mode='max',
                                 save_weights_only=True)
    callbacks_list = [checkpoint]

    history = model.fit(
        DataGenerator(feature_train, uv_tex_feature_train, distribution_label_train, batch_size=args.batch_size,
                      shuffle=True, train=True),
        steps_per_epoch=len(feature_train) // args.batch_size,
        epochs=args.epoch,
        validation_data=DataGenerator(feature_test, uv_tex_feature_test, distribution_label_test, batch_size=args.batch_size,
                      shuffle=True, train=False),
        verbose=1,
        callbacks=callbacks_list,
        use_multiprocessing=False
        )