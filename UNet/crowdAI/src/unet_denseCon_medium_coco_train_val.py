"""
Original code based from ternaus/kaggle_dstl_submission

Modified by Jasmine Hu to 
1. Have densor concatenation connections inspired by DenseNet
2. Take RGB input
3. Calculate validation/test performance in addition to training
4. Reduce channel widths

"""
from __future__ import division

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D,AveragePooling2D

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

import tensorflow as tf
from keras.models import model_from_json
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))
img_rows = 112
img_cols = 112


smooth = 1e-12

num_channels = 3
num_mask_channels = 1
random.seed(0)


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
    conv1a = BatchNormalization(mode=0, axis=1)(inputs)
    conv1a = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv1a)
    conv1a = keras.layers.advanced_activations.ELU()(conv1a)
    conv1b = BatchNormalization(mode=0, axis=1)(conv1a)
    conv1b = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv1b)
    conv1b = keras.layers.advanced_activations.ELU()(conv1b)
    conc1 = merge([conv1a,conv1b], mode = 'concat', concat_axis = 1)
    pool1 = keras.layers.AveragePooling2D(pool_size=(2, 2),dim_ordering='th')(conc1)

    conv2a = BatchNormalization(mode=0, axis=1)(pool1)
    conv2a = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv2a)
    conv2a = keras.layers.advanced_activations.ELU()(conv2a)
    conv2b = BatchNormalization(mode=0, axis=1)(conv2a)
    conv2b = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv2b)
    conc2 = merge([conv2a,conv2b], mode = 'concat', concat_axis = 1)
    pool2 = Convolution2D(24,1,1,border_mode='same',init='he_uniform',dim_ordering='th')(conc2)
    pool2 = AveragePooling2D(pool_size=(2, 2),dim_ordering='th')(conc2)

    conv3a = BatchNormalization(mode=0, axis=1)(pool2)
    conv3a = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv3a)
    conv3a = keras.layers.advanced_activations.ELU()(conv3a)
    conv3b = BatchNormalization(mode=0, axis=1)(conv3a)
    conv3b = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv3b)
    conv3b = keras.layers.advanced_activations.ELU()(conv3b)
    conc3 = merge([conv3a,conv3b], mode = 'concat', concat_axis = 1)
    pool3 = Convolution2D(48,1,1,border_mode='same',init='he_uniform',dim_ordering='th')(conc3)
    pool3 = AveragePooling2D(pool_size=(2, 2),dim_ordering='th')(conc3)
    
    conv4a = BatchNormalization(mode=0, axis=1)(pool3)
    conv4a = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv4a)
    conv4a = keras.layers.advanced_activations.ELU()(conv4a)
    conv4b = BatchNormalization(mode=0, axis=1)(conv4a)
    conv4b = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv4b)
    conv4b = keras.layers.advanced_activations.ELU()(conv4b)
    conc4 = merge([conv4a,conv4b], mode = 'concat', concat_axis = 1)
    pool4 = Convolution2D(96,1,1,border_mode='same',init='he_uniform',dim_ordering='th')(conc4)
    pool4 = AveragePooling2D(pool_size=(2, 2),dim_ordering='th')(conv4b)

  
    conv5a = BatchNormalization(mode=0, axis=1)(pool4)
    conv5a = Convolution2D(192, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv5a)
    conv5a = keras.layers.advanced_activations.ELU()(conv5a)
    conv5b = BatchNormalization(mode=0, axis=1)(conv5a)
    conv5b = Convolution2D(192, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv5b)
    conv5b = keras.layers.advanced_activations.ELU()(conv5b)
    conc5 = merge([conv5a,conv5b], mode = 'concat', concat_axis = 1)


    up6 = merge([UpSampling2D(size=(2, 2),dim_ordering='th')(conc5), conc4], mode='concat', concat_axis=1)
    conv6a = BatchNormalization(mode=0, axis=1)(up6)
    conv6a = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv6a)
    conv6a = keras.layers.advanced_activations.ELU()(conv6a)
    up6a = merge([up6, conv6a], mode = 'concat', concat_axis = 1)
    conv6b = BatchNormalization(mode=0, axis=1)(up6a)
    conv6b = Convolution2D(96, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv6b)
    conv6b = keras.layers.advanced_activations.ELU()(conv6b)
    conc6 = merge([up6a,conv6b], mode = 'concat', concat_axis = 1)
    conc6 = Convolution2D(96,1,1,border_mode='same',init='he_uniform',dim_ordering='th')(conc6)

    
    up7 = merge([UpSampling2D(size=(2, 2),dim_ordering='th')(conc6), conc3], mode='concat', concat_axis=1)
    conv7a = BatchNormalization(mode=0, axis=1)(up7)
    conv7a = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv7a)
    conv7a = keras.layers.advanced_activations.ELU()(conv7a)
    up7a = merge([up7, conv7a], mode = 'concat', concat_axis = 1)
    conv7b = BatchNormalization(mode=0, axis=1)(up7a)
    conv7b = Convolution2D(48, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv7b)
    conv7b = keras.layers.advanced_activations.ELU()(conv7b)
    conc7 = merge([up7a,conv7b], mode = 'concat', concat_axis = 1)
    conc7 = Convolution2D(48,1,1,border_mode='same',init='he_uniform',dim_ordering='th')(conc7)


    up8 = merge([UpSampling2D(size=(2, 2),dim_ordering='th')(conc7), conc2], mode='concat', concat_axis=1)
    conv8a = BatchNormalization(mode=0, axis=1)(up8)
    conv8a = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv8a)
    conv8a = keras.layers.advanced_activations.ELU()(conv8a)
    up8a = merge([up8, conv8a], mode = 'concat', concat_axis = 1)
    conv8b = BatchNormalization(mode=0, axis=1)(up8a)
    conv8b = Convolution2D(24, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv8b)
    conv8b = keras.layers.advanced_activations.ELU()(conv8b)
    conc8 = merge([up8a,conv8b], mode = 'concat', concat_axis = 1)
    conc8 = Convolution2D(24,1,1,border_mode='same',init='he_uniform',dim_ordering='th')(conc8)


    up9 = merge([UpSampling2D(size=(2, 2),dim_ordering='th')(conc8), conc1], mode='concat', concat_axis=1)
    conv9a = BatchNormalization(mode=0, axis=1)(up9)
    conv9a = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv9a)
    conv9a = keras.layers.advanced_activations.ELU()(conv9a)
    up9a = merge([up9, conv9a], mode = 'concat', concat_axis = 1)
    conv9b = BatchNormalization(mode=0, axis=1)(up9a)
    conv9b = Convolution2D(12, 3, 3, border_mode='same', init='he_uniform',dim_ordering='th')(conv9b)
    conv9b = keras.layers.advanced_activations.ELU()(conv9b)
    conc9 = merge([up9a,conv9b], mode = 'concat', concat_axis = 1)
    
    crop9 = Cropping2D(cropping=((16, 16), (16, 16)),dim_ordering='th')(conc9)
    conv9 = BatchNormalization(mode=0, axis=1)(crop9)
    conv9 = Convolution2D(12, 3, 3, border_mode='same',init='he_uniform',dim_ordering='th')(conv9)
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

def form_batch_val(X, y, batch_size,horizontal_flip=True,vertical_flip=True,swap_axis=True):
    X_batch = np.zeros((batch_size, num_channels, 112, 112))
    y_batch = np.zeros((batch_size, num_mask_channels, 80, 80))
    X_height = X.shape[2]
    X_width = X.shape[3]

    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)

        #random_image = random.randint(0, X.shape[0] - 1) or (0,909) or (910,1820-1)
        random_image=i*5

        y_batch[i] = y[random_image, :, random_height+16: random_height + 80+16, random_width+16 : random_width + 80+16]
        X_batch[i] = np.array(X[random_image, :, random_height: random_height + 112, random_width: random_width + 112])

        xb=X_batch[i]
        yb=y_batch[i]

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

# form validation/test batches. 

def batch_generator_val(X, y, batch_size, horizontal_flip=False, vertical_flip=False, swap_axis=False):
    while True:
        X_batch, y_batch = form_batch_val(X, y, batch_size)

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
    json_name = 'architecture_densenet' + cross + '.json'
    weight_name = 'model_weights_densenet' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def save_history(history, suffix):
    filename = 'history/history_' + suffix + '.csv'
    pd.DataFrame(history.history).to_csv(filename, index=False)


def read_model(cross=''):
    json_name = 'architecture_unet1' + cross + '.json'
    weight_name = 'model_weights_unet1' + cross + '.h5'
    model = model_from_json(open(os.path.join('../src/cache', json_name)).read())
    model.load_weights(os.path.join('../src/cache', weight_name))
    return model


if __name__ == '__main__':
    data_path = '../data'
    now = datetime.datetime.now()

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet0()

    print('[{}] Reading train...'.format(str(datetime.datetime.now())))
    f = h5py.File(os.path.join(data_path, 'train_coco.h5'), 'r')

    X_train = f['train']

    y_train = np.array(f['train_mask_coco'])[:, 0]
    y_train = np.expand_dims(y_train, 1)
    print(y_train.shape)

    train_ids = np.array(f['train_ids'])


    print('[{}] Reading val...'.format(str(datetime.datetime.now())))
    fv = h5py.File(os.path.join(data_path, 'val_coco.h5'), 'r')

    X_val = fv['val']

    y_val = np.array(fv['val_mask_coco'])[:, 0]
    y_val = np.expand_dims(y_val, 1)
    print(y_val.shape)

    val_ids = np.array(fv['val_ids'])


    batch_size = 128
    nb_epoch = 4

    history = History()
    callbacks = [
        history,
    ]

    suffix = 'buildings_3_densenet_medium_coco_eval'+"{batch}_{epoch}".format(batch=batch_size,epoch=nb_epoch)
    X_val_ev, y_val_ev = form_batch_val(X_val,y_val,320)

    model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    model.fit_generator(batch_generator(X_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
                        #validation_data=batch_generator(X_val,y_val,batch_size, horizontal_flip=True, vertical_flip=True, swap_axis = True),
                        #validation_steps=1,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        samples_per_epoch=batch_size * 25,
                        callbacks=callbacks,
                        nb_worker=24
                        )

   
    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
    
    # Evaluation on validation/test set.
    ev=model.evaluate(X_val_ev,y_val_ev)
    print(ev)

    save_history(history, suffix)

    f.close()
