"""
Original code based from ternaus/kaggle_dstl_submission

Modified by Jasmine Hu to 
1. Make predictions on CrowdAI mapping challenge val/test sets
2. Directly produces images.

"""

from __future__ import division

import os
from tqdm import tqdm
import pandas as pd
import extra_functions
import shapely.geometry
from numba import jit

from keras.models import model_from_json
import numpy as np
import datetime
import random
import h5py
import matplotlib.pyplot as plt

random.seed(0)


   

#suffix = 'buildings_3_unet1_medium_coco'+"{batch}_{epoch}".format(batch=128,epoch=4)

#modelCross="{batch}_{epoch}_{suffix}".format(batch=128, epoch=4, suffix=suffix))

def read_model(cross=''):
    json_name = 'architecture_unet1128_4_buildings_3_unet1_medium_coco128_4' + cross + '.json'
    weight_name = 'model_weights_unet1128_4_buildings_3_unet1_medium_coco128_4' + cross + '.h5'
    model = model_from_json(open(os.path.join('../src/cache', json_name)).read())
    model.load_weights(os.path.join('../src/cache', weight_name))
    return model

model = read_model()

#sample = pd.read_csv('../data/sample_submission.csv')

data_path = '../data'
num_channels = 3
num_mask_channels = 1
threashold = 0.5

#three_band_path = os.path.join(data_path, 'three_band')

#train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
#gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
#shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))

#test_ids = shapes.loc[~shapes['image_id'].isin(train_wkt['ImageId'].unique()), 'image_id']

result = []
test_ids=[]

print('[{}] Reading val...'.format(str(datetime.datetime.now())))
fv = h5py.File(os.path.join(data_path, 'val_coco.h5'), 'r')

X_val = fv['val']

y_val = np.array(fv['val_mask_coco'])[:, 0]
y_val = np.expand_dims(y_val, 1)
print(y_val.shape)

val_ids = np.array(fv['val_ids'])


batch_size = 100
img_rows=112
img_cols=112


def form_batch(X, y, batch_size):
    X_batch = np.zeros((batch_size, num_channels, img_rows, img_cols))
    y_batch = np.zeros((batch_size, num_mask_channels, img_rows, img_cols))
    X_height = X.shape[2]
    X_width = X.shape[3]

    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1) # or 50 to know where the cropping starts to test
        random_height = random.randint(0, X_height - img_rows - 1) # or 50 to know where the cropping starts to test

        random_image = random.randint(0, X.shape[0] - 1)
        test_ids.append(random_image)

        y_batch[i] = y[i, :, random_height: random_height + img_rows, random_width: random_width + img_cols]
        X_batch[i] = np.array(X[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols])
    return X_batch, y_batch

X_batch,y_batch=form_batch(X_val,y_val,batch_size)

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

@jit
def mask2poly(predicted_mask, threashold, x_scaler, y_scaler):
    polygons = extra_functions.mask2polygons_layer(predicted_mask[0] > threashold, epsilon=0, min_area=5)

    polygons = shapely.affinity.scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
    return shapely.wkt.dumps(polygons.buffer(2.6e-5))

from PIL import Image

for i in np.arange(batch_size):
    test_id=test_ids[i]
    print(test_id)

    image = X_batch[i]

    #image = extra_functions.read_image_16(image_id)

    #H = image.shape[1]
    #W = image.shape[2]

    #x_max, y_min = extra_functions._get_xmax_ymin(image_id)

    predicted_mask = extra_functions.make_prediction_cropped(model, image, initial_size=(112, 112),
                                                             final_size=(112-32, 112-32),
                                                             num_masks=num_mask_channels, num_channels=num_channels)

    image_v = flip_axis(image, 1)
    predicted_mask_v = extra_functions.make_prediction_cropped(model, image_v, initial_size=(112, 112),
                                                               final_size=(112 - 32, 112 - 32),
                                                               num_masks=1,
                                                               num_channels=num_channels)

    image_h = flip_axis(image, 2)
    predicted_mask_h = extra_functions.make_prediction_cropped(model, image_h, initial_size=(112, 112),
                                                               final_size=(112 - 32, 112 - 32),
                                                               num_masks=1,
                                                               num_channels=num_channels)

    image_s = image.swapaxes(1, 2)
    predicted_mask_s = extra_functions.make_prediction_cropped(model, image_s, initial_size=(112, 112),
                                                               final_size=(112 - 32, 112 - 32),
                                                               num_masks=1,
                                                               num_channels=num_channels)

    mask = np.power(predicted_mask *
                        flip_axis(predicted_mask_v, 1) *
                        flip_axis(predicted_mask_h, 2) *
                        predicted_mask_s.swapaxes(1, 2), 0.25)

    print("************")
    print(image.shape)
    image2 = 1.0 / 255.0 * np.moveaxis(image,0,-1)
    mask2 = np.moveaxis(np.clip(mask, 0.0, 1.0),0,-1) * 0.6 + 0.0
    mask_final=np.zeros((112,112,3))
    mask_final[:,:,0]=mask2[:,:,0] * 0.8;
    mask_final[:,:,1]=mask2[:,:,0] * 0.5;
    mask_final[:,:,2]=mask2[:,:,0] * 0.2;
    print(image2.shape, mask_final.shape)
    plt.imsave("%05d-image-%05d.png" % (i, test_id), image2)
    plt.imsave("%05d-mask-%05d.png" % (i, test_id), mask_final)
    plt.imsave("%05d-combined-%05d.png" % (i, test_id), np.clip(image2 + mask_final, 0.0, 1.0))

    #x_scaler, y_scaler = extra_functions.get_scalers(H, W, x_max, y_min)

    # mask_channel = 0
    #result += [(test_id, mask_channel + 1, mask2poly(new_mask, threashold, 1, 1))]

# submission = pd.DataFrame(result, columns=['ImageId', 'ClassType', 'MultipolygonWKT'])

# sample = sample.drop('MultipolygonWKT', 1)
#submission = sample.merge(submission, on=['ImageId', 'ClassType'], how='left').fillna('MULTIPOLYGON EMPTY')
# submission.to_csv('temp_building_coco.csv', index=False)
