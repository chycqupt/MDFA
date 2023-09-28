from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D

from keras.utils import layer_utils
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import warnings
from tensorflow.keras.models import Model


def VGG16(include_top=True, weights='vggface',
          input_tensor=None, input_shape=(224, 224, 3),
          pooling=None,
          classes=2622):

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(
        x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(
        x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(
        x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(
        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, name='fc6')(x)
        x = Activation('relu', name='fc6/relu')(x)
        x = Dense(4096, name='fc7')(x)
        x = Activation('relu', name='fc7/relu')(x)
        x = Dense(classes, name='fc8')(x)
        x = Activation('softmax', name='fc8/softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

            # Ensure that the model takes into account
            # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        #inputs = get_source_inputs(input_tensor)
        inputs = img_input
    else:
        inputs = img_input
        # Create model.
    model = Model(inputs, x, name='vggface_vgg16')  # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = 'rcmalli_vggface_tf_vgg16.h5'
        else:
            weights_path = 'rcmalli_vggface_tf_notop_vgg16.h5'
        model.load_weights(weights_path, by_name=True)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='pool5')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc6')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def resnet_identity_block(input_tensor, kernel_size, filters, stage, block,
                          bias=False):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, use_bias=bias,
               padding='same', name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def resnet_conv_block(input_tensor, kernel_size, filters, stage, block,
                      strides=(2, 2), bias=False):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=bias,
                      name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "/bn")(
        shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def RESNET50(include_top=True, weights='vggface',
             input_tensor=None, input_shape=(224, 224, 3),
             pooling=None,
             classes=8631):

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(
        64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
        name='conv1/7x7_s2')(img_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = layers.Dropout(0.5)(x)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = layers.Dropout(0.5)(x)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)
    x = layers.Dropout(0.5)(x)

    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = layers.Dropout(0.5)(x)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = layers.Dropout(0.5)(x)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = layers.Dropout(0.5)(x)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)
    x = layers.Dropout(0.5)(x)

    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = layers.Dropout(0.5)(x)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = layers.Dropout(0.5)(x)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = layers.Dropout(0.5)(x)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = layers.Dropout(0.5)(x)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = layers.Dropout(0.5)(x)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)
    x = layers.Dropout(0.5)(x)

    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = layers.Dropout(0.5)(x)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = layers.Dropout(0.5)(x)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)
    x = layers.Dropout(0.5)(x)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='classifier')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = img_input
        #inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vggface_resnet50')

    # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = 'rcmalli_vggface_tf_resnet50.h5'
        else:
            weights_path = 'rcmalli_vggface_tf_notop_resnet50.h5'
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='classifier')
                layer_utils.convert_dense_weights_data_format(dense, shape,
                                                              'channels_first')

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    elif weights is not None:
        model.load_weights(weights)

    return model




