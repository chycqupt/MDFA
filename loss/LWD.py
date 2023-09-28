import tensorflow as tf
import tensorflow.keras.backend as K

def complex_loss(y_true, y_pred):
    y_pred = K.cast(y_pred, dtype=tf.float32)  # None*5
    y_true = K.cast(y_true, dtype=tf.float32)  # None*5
    loss_mse = K.mean(K.square(y_pred - y_true), axis=-1)
    mean_pred, var_pred = tf.nn.moments(y_pred, axes=0)
    mean_true, var_true = tf.nn.moments(y_true, axes=0)
    temp_mean = K.reshape((mean_pred - mean_true), (-1, 1))
    temp_std = K.reshape((var_pred - var_true), (-1, 1))
    mean_loss = K.batch_dot(temp_mean, temp_mean)
    std_loss = K.batch_dot(temp_std, temp_std)
    loss = K.mean(K.square(mean_loss + std_loss), axis=0) + loss_mse
    return loss