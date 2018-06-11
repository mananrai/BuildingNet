"""
Original code based from ternaus/kaggle_dstl_submission

Modified by Jasmine Hu to 
1. Read crowdAI mapping challenge data in MS COCO annotation
2. Caches train data for future trainings.

"""

from __future__ import division

import os
import pandas as pd
from tqdm import tqdm
import h5py
import numpy as np

from mrcnn.dataset import MappingChallengeDataset

data_path = './data'

# train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
# gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
# 
# shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))


def cache_train_16():



    # Load training dataset
    dataset_train = MappingChallengeDataset()
    dataset_train.load_dataset(dataset_dir=os.path.join("data", "train"), load_small=True)
    dataset_train.prepare()
    image_ids=np.copy(dataset_train.image_ids)
    
    
    num_train = len(image_ids)
    img = dataset_train.load_image(image_ids[0])
    img_trans=np.moveaxis(img,-1,0)

    image_rows = img_trans.shape[1]
    image_cols = img_trans.shape[2]
    num_channels = img_trans.shape[0]
    num_mask_channels = 1


    f = h5py.File(os.path.join(data_path, 'train_coco.h5'), 'w')
    imgs = f.create_dataset('train', (num_train, num_channels, image_rows, image_cols), dtype=np.float16)

    imgs_mask = f.create_dataset('train_mask_coco', (num_train, num_mask_channels, image_rows, image_cols), dtype=np.uint8)

    ids = []

    i = 0

    for image_id in image_ids:
        img = dataset_train.load_image(image_id)
        img_trans=np.moveaxis(img,-1,0)
        mask, class_ids = dataset_train.load_mask(image_id)

        if mask.shape[0] == 0:
            mask = np.zeros((num_mask_channels, image_rows, image_cols))

        flat_mask = np.sum(mask,axis=2,keepdims=True)
        flat_mask_trans=np.moveaxis(flat_mask,-1,0)
       
        print("img id %d maskshape %s" % (image_id, mask.shape))
        print("img id %d maskshape %s maskmax %d" % (image_id, mask.shape, np.amax(mask)))
        imgs[i] = img_trans
        imgs_mask[i] = flat_mask_trans
        ids += [image_id]
        i += 1

        #print("new stuff. image id %d" % image_id)
        # print(image_id)
        # print(img.shape)
        #print(img_trans.shape)
        #print(mask.shape)
        #print(class_ids)
        #print(flat_mask_trans.shape)
        #print(np.amax(flat_mask))
        
    f['train_ids'] = np.array(ids).astype('|S9')

    f.close()
    print("there are %d number images" % i)




    # print('num_train_images =', train_wkt['ImageId'].nunique())

    # train_shapes = shapes[shapes['image_id'].isin(train_wkt['ImageId'].unique())]

    # min_train_height = train_shapes['height'].min()
    # min_train_width = train_shapes['width'].min()

    # num_train = train_shapes.shape[0]

    # image_rows = min_train_height
    # image_cols = min_train_width

    # num_channels = 3

    # num_mask_channels = 10

    # f = h5py.File(os.path.join(data_path, 'train_coco.h5'), 'w')

    # imgs = f.create_dataset('train', (num_train, num_channels, image_rows, image_cols), dtype=np.float16)
    # imgs_mask = f.create_dataset('train_mask', (num_train, num_mask_channels, image_rows, image_cols), dtype=np.uint8)

    # ids = []

    # i = 0
    # # 
    # for image_id in tqdm(sorted(train_wkt['ImageId'].unique())):
    #     image = extra_functions.read_image_16(image_id)
    #     _, height, width = image.shape

    #     imgs[i] = image[:, :min_train_height, :min_train_width]

    #     imgs_mask[i] = extra_functions.generate_mask(image_id,
    #                                                  height,
    #                                                  width,
    #                                                  num_mask_channels=num_mask_channels,
    #                                                  train=train_wkt)[:, :min_train_height, :min_train_width]

    #     ids += [image_id]
    #     i += 1

    # # fix from there: https://github.com/h5py/h5py/issues/441
    # f['train_ids'] = np.array(ids).astype('|S9')

    # f.close()


if __name__ == '__main__':
    cache_train_16()

