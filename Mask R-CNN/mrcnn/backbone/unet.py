from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D

from keras import backend as K

import keras
import h5py

from keras.layers.normalization import BatchNormalization

img_rows = 112
img_cols = 112

smooth = 1e-12

num_channels = 3
num_mask_channels = 1

# def unet_graph(inputs):
#     # inputs = Input((num_channels, img_rows, img_cols))
#     conv1 = Convolution2D(6, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(inputs)
#     conv1 = BatchNormalization(mode=0, axis=1)(conv1)
#     conv1 = keras.layers.advanced_activations.ELU()(conv1)
#     conv1 = Convolution2D(6, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv1)
#     conv1 = BatchNormalization(mode=0, axis=1)(conv1)
#     conv1 = keras.layers.advanced_activations.ELU()(conv1)
#     C1 = pool1 = MaxPooling2D(pool_size=(2, 2),dim_ordering='th')(conv1)
#
#     conv2 = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(pool1)
#     conv2 = BatchNormalization(mode=0, axis=1)(conv2)
#     conv2 = keras.layers.advanced_activations.ELU()(conv2)
#     conv2 = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv2)
#     conv2 = BatchNormalization(mode=0, axis=1)(conv2)
#     conv2 = keras.layers.advanced_activations.ELU()(conv2)
#     C2 = pool2 = MaxPooling2D(pool_size=(2, 2),dim_ordering='th')(conv2)
#
#     conv3 = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(pool2)
#     conv3 = BatchNormalization(mode=0, axis=1)(conv3)
#     conv3 = keras.layers.advanced_activations.ELU()(conv3)
#     conv3 = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv3)
#     conv3 = BatchNormalization(mode=0, axis=1)(conv3)
#     conv3 = keras.layers.advanced_activations.ELU()(conv3)
#     C3 = pool3 = MaxPooling2D(pool_size=(2, 2),dim_ordering='th')(conv3)
#
#     conv4 = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(pool3)
#     conv4 = BatchNormalization(mode=0, axis=1)(conv4)
#     conv4 = keras.layers.advanced_activations.ELU()(conv4)
#     conv4 = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv4)
#     conv4 = BatchNormalization(mode=0, axis=1)(conv4)
#     conv4 = keras.layers.advanced_activations.ELU()(conv4)
#     C4 = pool4 = MaxPooling2D(pool_size=(2, 2),dim_ordering='th')(conv4)
#
#     conv5 = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(pool4)
#     conv5 = BatchNormalization(mode=0, axis=1)(conv5)
#     conv5 = keras.layers.advanced_activations.ELU()(conv5)
#     conv5 = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv5)
#     conv5 = BatchNormalization(mode=0, axis=1)(conv5)
#     C5 = conv5 = keras.layers.advanced_activations.ELU()(conv5)
#
#     up6 = merge([UpSampling2D(size=(2, 2),dim_ordering='th')(conv5), conv4], mode='concat', concat_axis=1)
#     conv6 = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(up6)
#     conv6 = BatchNormalization(mode=0, axis=1)(conv6)
#     conv6 = keras.layers.advanced_activations.ELU()(conv6)
#     conv6 = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv6)
#     conv6 = BatchNormalization(mode=0, axis=1)(conv6)
#     P5 = conv6 = keras.layers.advanced_activations.ELU()(conv6)
#
#     up7 = merge([UpSampling2D(size=(2, 2),dim_ordering='th')(conv6), conv3], mode='concat', concat_axis=1)
#     conv7 = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(up7)
#     conv7 = BatchNormalization(mode=0, axis=1)(conv7)
#     conv7 = keras.layers.advanced_activations.ELU()(conv7)
#     conv7 = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv7)
#     conv7 = BatchNormalization(mode=0, axis=1)(conv7)
#     P4 = conv7 = keras.layers.advanced_activations.ELU()(conv7)
#
#     up8 = merge([UpSampling2D(size=(2, 2),dim_ordering='th')(conv7), conv2], mode='concat', concat_axis=1)
#     conv8 = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(up8)
#     conv8 = BatchNormalization(mode=0, axis=1)(conv8)
#     conv8 = keras.layers.advanced_activations.ELU()(conv8)
#     conv8 = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv8)
#     conv8 = BatchNormalization(mode=0, axis=1)(conv8)
#     P3 = conv8 = keras.layers.advanced_activations.ELU()(conv8)
#
#     up9 = merge([UpSampling2D(size=(2, 2),dim_ordering='th')(conv8), conv1], mode='concat', concat_axis=1)
#     conv9 = Convolution2D(6, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(up9)
#     conv9 = BatchNormalization(mode=0, axis=1)(conv9)
#     conv9 = keras.layers.advanced_activations.ELU()(conv9)
#     conv9 = Convolution2D(6, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv9)
#     crop9 = Cropping2D(cropping=((16, 16), (16, 16)),dim_ordering='th')(conv9)
#     conv9 = BatchNormalization(mode=0, axis=1)(crop9)
#     P2 = conv9 = keras.layers.advanced_activations.ELU()(conv9)
#
#     conv10 = Convolution2D(num_mask_channels, 1, 1, activation='sigmoid',dim_ordering='th')(conv9)
#
#     # model = Model(input=inputs, output=conv10)
#
#     return [C1, C2, C3, C4, C5, P2, P3, P4, P5]

def unet_graph(input_tensor):
    # inputs = Input((num_channels, img_rows, img_cols))
    skip1 = Convolution2D(12, 1, 1, init='he_uniform')(input_tensor)
    conv1 = Convolution2D(12, 3, 3, init='he_uniform', dim_ordering='th')(input_tensor)
    conv1 = BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    conv1 = Convolution2D(12, 3, 3, init='he_uniform', dim_ordering='th')(conv1)
    conv1 = BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = keras.layers.Add()([conv1, skip1])
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), dim_ordering='th')(conv1)

    skip2 = Convolution2D(24, 1, 1, border_mode='same', init='he_uniform', dim_ordering='th')(pool1)
    conv2 = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(pool1)
    conv2 = BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    conv2 = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(conv2)
    conv2 = BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = keras.layers.Add()([conv2, skip2])
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), dim_ordering='th')(conv2)

    skip3 = Convolution2D(48, 1, 1, border_mode='same', init='he_uniform', dim_ordering='th')(pool2)
    conv3 = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(pool2)
    conv3 = BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    conv3 = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(conv3)
    conv3 = BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = keras.layers.Add()([conv3, skip3])
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), dim_ordering='th')(conv3)

    skip4 = Convolution2D(96, 1, 1, border_mode='same', init='he_uniform', dim_ordering='th')(pool3)
    conv4 = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(pool3)
    conv4 = BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    conv4 = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(conv4)
    conv4 = BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = keras.layers.Add()([conv4, skip4])
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), dim_ordering='th')(conv4)

    skip5 = Convolution2D(192, 1, 1, border_mode='same', init='he_uniform', dim_ordering='th')(pool4)
    conv5 = Convolution2D(192, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(pool4)
    conv5 = BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)
    conv5 = Convolution2D(192, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(conv5)
    conv5 = BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = keras.layers.Add()([conv5, skip5])
    conv5 = keras.layers.advanced_activations.ELU()(conv5)

    up6 = merge([UpSampling2D(size=(2, 2), dim_ordering='th')(conv5), conv4], mode='concat', concat_axis=1)
    skip6 = Convolution2D(96, 1, 1, border_mode='same', init='he_uniform', dim_ordering='th')(up6)
    conv6 = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(up6)
    conv6 = BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)
    conv6 = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(conv6)
    conv6 = BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = keras.layers.Add()([conv6, skip6])
    C1 = conv6 = keras.layers.advanced_activations.ELU()(conv6)

    up7 = merge([UpSampling2D(size=(2, 2), dim_ordering='th')(conv6), conv3], mode='concat', concat_axis=1)
    skip7 = Convolution2D(48, 1, 1, border_mode='same', init='he_uniform', dim_ordering='th')(up7)
    conv7 = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(up7)
    conv7 = BatchNormalization(mode=0, axis=1)(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)
    conv7 = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(conv7)
    conv7 = BatchNormalization(mode=0, axis=1)(conv7)
    conv7 = keras.layers.Add()([conv7, skip7])
    C2 = conv7 = keras.layers.advanced_activations.ELU()(conv7)

    up8 = merge([UpSampling2D(size=(2, 2), dim_ordering='th')(conv7), conv2], mode='concat', concat_axis=1)
    skip8 = Convolution2D(24, 1, 1, border_mode='same', init='he_uniform', dim_ordering='th')(up8)
    conv8 = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(up8)
    conv8 = BatchNormalization(mode=0, axis=1)(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)
    conv8 = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(conv8)
    conv8 = BatchNormalization(mode=0, axis=1)(conv8)
    conv8 = keras.layers.Add()([conv8, skip8])
    C3 = conv8 = keras.layers.advanced_activations.ELU()(conv8)

    up9 = merge([UpSampling2D(size=(2, 2), dim_ordering='th')(conv8), conv1], mode='concat', concat_axis=1)
    skip9 = Convolution2D(12, 1, 1, border_mode='same', init='he_uniform', dim_ordering='th')(up9)
    conv9 = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(up9)
    conv9 = BatchNormalization(mode=0, axis=1)(conv9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv9 = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform', dim_ordering='th')(conv9)
    conv9 = keras.layers.Add()([conv9, skip9])
    crop9 = Cropping2D(cropping=((16, 16), (16, 16)), dim_ordering='th')(conv9)
    conv9 = BatchNormalization(mode=0, axis=1)(crop9)
    C4 = conv9 = keras.layers.advanced_activations.ELU()(conv9)

    C5 = conv10 = Convolution2D(num_mask_channels, 1, 1, activation='sigmoid', dim_ordering='th')(conv9)

    return [C1, C2, C3, C4, C5]