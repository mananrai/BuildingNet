import matplotlib
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os
import json
import copy
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


data_directory = "data/"
annotation_file_template = "{}/{}/annotation{}.json"

TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotation.json"
TRAIN_ANNOTATIONS_SMALL_PATH = "data/train/annotation-small.json"

VAL_IMAGES_DIRECTORY = "data/val/images"
VAL_ANNOTATIONS_PATH = "data/val/annotation.json"
VAL_ANNOTATIONS_SMALL_PATH = "data/val/annotation-small.json"

TEST_IMAGES_DIRECTORY = "data/test/images"
TEST_ANNOTATIONS_PATH = "data/test/annotation.json"
TEST_ANNOTATIONS_SMALL_PATH = "data/test/annotation-small.json"


coco = COCO(VAL_ANNOTATIONS_PATH)


category_ids = coco.loadCats(coco.getCatIds())
print(category_ids)
image_ids = coco.getImgIds(catIds=coco.getCatIds())

NUMBER_OF_VAL_IMAGES = len(image_ids)
NUMBER_OF_TEST_IMAGES = NUMBER_OF_VAL_IMAGES // 2
print(NUMBER_OF_VAL_IMAGES)
print(NUMBER_OF_TEST_IMAGES)
input("Enter...")

with open(VAL_ANNOTATIONS_PATH, 'r+') as infile:
    data = json.load(infile)
    print(data.keys())
    # print(data["annotations"][0])

    test_data = copy.deepcopy(data)
    test_data["images"] = []
    test_data["annotations"] = []

    for i in range(NUMBER_OF_TEST_IMAGES):
        random_image_id = random.choice(image_ids)
        img = coco.loadImgs(random_image_id)[0]
        annotation_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(annotation_ids)
        # print(data["annotations"].index(annotations[0]))
        # print(test_data["annotations"].index(annotations[0]))
        # print(len(annotations))
        # print(len(data["annotations"]))
        # print(len(test_data["annotations"]))
        data["annotations"] = [a for a in data["annotations"] if a not in annotations]
        test_data["annotations"] += annotations
        # print(data["annotations"].index(annotations[0]))
        # print(test_data["annotations"].index(annotations[0]))
        # print(len(data["annotations"]))
        # print(len(test_data["annotations"]))

        # print(data["images"].index(random_image_id))
        # print(test_data["images"].index(random_image_id))
        # print(len(data["images"]))
        # print(len(test_data["images"]))
        data["images"].remove(img)
        test_data["images"].append(img)
        image_ids.remove(random_image_id)
        # print(data["images"].index(random_image_id))
        # print(test_data["images"].index(random_image_id))
        # print(len(data["images"]))
        # print(len(test_data["images"]))

        source_image_path = os.path.join(VAL_IMAGES_DIRECTORY, img["file_name"])
        destination_image_path = os.path.join(TEST_IMAGES_DIRECTORY, img["file_name"])
        os.rename(source_image_path, destination_image_path)

with open(VAL_ANNOTATIONS_PATH, 'w') as outfile1:
    json.dump(data, outfile1)

with open(TEST_ANNOTATIONS_PATH, 'w') as outfile2:
    json.dump(test_data, outfile2)
    # print(data.keys())
    # input("Enter...")
    # json.dump(data, outfile)


# json.dump(data, infile)
# with open(TEST_ANNOTATIONS_PATH, 'w') as outfile:
#     json.dump(test_data, outfile)
#     # print(data.keys())
#     # input("Enter...")
#     # json.dump(data, outfile)

# # For this demonstration, we will randomly choose an image_id
# random_image_id = random.choice(image_ids)
# # Now that we have an image_id, we can load its corresponding object by doing :
# img = coco.loadImgs(random_image_id)[0]
#
#
# image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, img["file_name"])
# I = io.imread(image_path)
# plt.imshow(I)
#
# annotation_ids = coco.getAnnIds(imgIds=img['id'])
# annotations = coco.loadAnns(annotation_ids)
#
#
# # load and render the image
# plt.imshow(I); plt.axis('off')
# # Render annotations on top of the image
# coco.showAnns(annotations)
#
#
# from pycocotools import mask as cocomask
# rle = cocomask.frPyObjects(annotations[0]['segmentation'], img['height'], img['width'])
# m = cocomask.decode(rle)
# # m.shape has a shape of (300, 300, 1)
# # so we first convert it to a shape of (300, 300)
# m = m.reshape((img['height'], img['width']))
# plt.imshow(m)
