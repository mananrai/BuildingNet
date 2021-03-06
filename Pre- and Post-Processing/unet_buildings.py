"""
Original code based on Kaggle competition

Modified to take 3-channel input
"""
from __future__ import division

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D

from keras import backend as K

import keras
import h5py

from keras.layers.normalization import BatchNormalization


from keras.optimizers import Nadam
from keras.callbacks import History
import pandas as pd
from keras.backend import binary_crossentropy

import datetime
import os

import random
import threading

#-----Vivian added--------#
from skimage.morphology import binary_dilation, binary_erosion, dilation, rectangle, binary_opening
import numpy as np
#-----Vivian added--------#

from keras.models import model_from_json

img_rows = 112
img_cols = 112


smooth = 1e-12

num_channels = 3
num_mask_channels = 1


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


def get_unet0():
    inputs = Input((num_channels, img_rows, img_cols))
    conv1 = Convolution2D(6, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(inputs)
    conv1 = BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    conv1 = Convolution2D(6, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv1)
    conv1 = BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),dim_ordering='th')(conv1)

    conv2 = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(pool1)
    conv2 = BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    conv2 = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv2)
    conv2 = BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),dim_ordering='th')(conv2)

    conv3 = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(pool2)
    conv3 = BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    conv3 = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv3)
    conv3 = BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),dim_ordering='th')(conv3)

    conv4 = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(pool3)
    conv4 = BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    conv4 = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv4)
    conv4 = BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),dim_ordering='th')(conv4)

    conv5 = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(pool4)
    conv5 = BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)
    conv5 = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv5)
    conv5 = BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)

    up6 = merge([UpSampling2D(size=(2, 2),dim_ordering='th')(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(up6)
    conv6 = BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)
    conv6 = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv6)
    conv6 = BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)

    up7 = merge([UpSampling2D(size=(2, 2),dim_ordering='th')(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(up7)
    conv7 = BatchNormalization(mode=0, axis=1)(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)
    conv7 = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv7)
    conv7 = BatchNormalization(mode=0, axis=1)(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)

    up8 = merge([UpSampling2D(size=(2, 2),dim_ordering='th')(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(up8)
    conv8 = BatchNormalization(mode=0, axis=1)(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)
    conv8 = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv8)
    conv8 = BatchNormalization(mode=0, axis=1)(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)

    up9 = merge([UpSampling2D(size=(2, 2),dim_ordering='th')(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(6, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(up9)
    conv9 = BatchNormalization(mode=0, axis=1)(conv9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv9 = Convolution2D(6, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv9)
    crop9 = Cropping2D(cropping=((16, 16), (16, 16)),dim_ordering='th')(conv9)
    conv9 = BatchNormalization(mode=0, axis=1)(crop9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)

    conv10 = Convolution2D(num_mask_channels, 1, 1, activation='sigmoid',dim_ordering='th')(conv9)

    model = Model(input=inputs, output=conv10)

    return model


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def form_batch(X, y, batch_size):
    X_batch = np.zeros((batch_size, num_channels, img_rows, img_cols))
    y_batch = np.zeros((batch_size, num_mask_channels, img_rows, img_cols))
    X_height = X.shape[2]
    X_width = X.shape[3]

    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)

        random_image = random.randint(0, X.shape[0] - 1)

        y_batch[i] = y[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols]
        X_batch[i] = np.array(X[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols])
    return X_batch, y_batch


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        yield threadsafe_iter(f(*a, **kw))
    return g


def batch_generator(X, y, batch_size, horizontal_flip=False, vertical_flip=False, swap_axis=False):
    while True:
        X_batch, y_batch = form_batch(X, y, batch_size)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = y_batch[i]

            if horizontal_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 1)
                    yb = flip_axis(yb, 1)

            if vertical_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 2)
                    yb = flip_axis(yb, 2)

            if swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(1, 2)
                    yb = yb.swapaxes(1, 2)

            X_batch[i] = xb
            y_batch[i] = yb

        yield X_batch, y_batch[:, :, 16:16 + img_rows - 32, 16:16 + img_cols - 32]


def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def save_history(history, suffix):
    filename = 'history/history_' + suffix + '.csv'
    pd.DataFrame(history.history).to_csv(filename, index=False)


def read_model(cross=''):
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    model = model_from_json(open(os.path.join('../src/cache', json_name)).read())
    model.load_weights(os.path.join('../src/cache', weight_name))
    return model


if __name__ == '__main__':
    data_path = '../data'
    now = datetime.datetime.now()

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet0()

    print('[{}] Reading train...'.format(str(datetime.datetime.now())))
    f = h5py.File(os.path.join(data_path, 'train_3.h5'), 'r')

    X_train = f['train']

    y_train = np.array(f['train_mask'])[:, 0]
    y_train = np.expand_dims(y_train, 1)
    print(y_train.shape)

    train_ids = np.array(f['train_ids'])

    batch_size = 128
    nb_epoch = 3

    history = History()
    callbacks = [
        history,
    ]

    suffix = 'buildings_3_'
    model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    model.fit_generator(batch_generator(X_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
                        nb_epoch=4,
                        verbose=1,
                        samples_per_epoch=batch_size * 25,
                        callbacks=callbacks,
                        nb_worker=24
                        )

    # changed from batch_size*400, nb_epoch = nb_epoch

    # do predict, then transfer to np array, then do skimage opening 

    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
    #save_history(history, suffix)

    f.close()
