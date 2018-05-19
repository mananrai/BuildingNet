# Mask RCNN
#
# This program contains the baseline code for training a Mask RCNN and making predictions.
#
# [Mask RCNN](https://arxiv.org/abs/1703.06870) model for the [crowdAI Mapping Challenge](https://www.crowdai.org/challenges/mapping-challenge).
#
# This code is adapted from the [Mask RCNN]() tensorflow implementation available here : [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN).
#
# Sources:
#   - This code is heavily inspired by the implementation by Sharada Mohanty for the CrowdAI Mapping Challenge
#       (https://www.crowdai.org/challenges/mapping-challenge) with minor changes
#   - Adapted from the tensorflow implementation available at https://github.com/matterport/Mask_RCNN
#   - Mask RCNN (https://arxiv.org/abs/1703.06870)
#

import os
import sys
import time
import numpy as np
import skimage.io

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
#
# A quick one liner to install the library
# !pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# import coco #a slightly modified version

from mrcnn.evaluate import build_coco_results, evaluate_coco
from mrcnn.dataset import MappingChallengeDataset
from mrcnn import visualize

import zipfile
import urllib.request
import shutil
import glob
import tqdm
import random

ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR, "data/" "pretrained_weights.h5")
LOGS_DIRECTORY = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "test", "images")

class BuildingsConfig(Config):
    """Configuration for training on data in MS COCO format.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Buildings"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 5

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building

    STEPS_PER_EPOCH=100
    VALIDATION_STEPS=20

    IMAGE_MAX_DIM=320
    IMAGE_MIN_DIM=320

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1

    IMAGES_PER_GPU = 5

    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building

    IMAGE_MAX_DIM = 320
    IMAGE_MIN_DIM = 320

    NAME = "crowdai-mapping-challenge"

def train():
    config = BuildingsConfig()
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOGS_DIRECTORY)

    # Load pretrained weights
    model_path = PRETRAINED_MODEL_PATH
    model.load_weights(model_path, by_name=True)

    # Load training dataset
    dataset_train = BuildingsDataset()
    dataset_train.load_dataset(dataset_dir=os.path.join("data", "train"), load_small=True)
    dataset_train.prepare()

    # Load validation dataset
    dataset_val = BuildingsDataset()
    val_coco = dataset_val.load_dataset(dataset_dir=os.path.join("data", "val"), load_small=True, return_coco=True)
    dataset_val.prepare()

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=40,
                layers='all')


def test(examples="one"):
    config = InferenceConfig()
    config.display()

    inference_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    model_path = PRETRAINED_MODEL_PATH

    # or if you want to use the latest trained model, you can use :
    # model_path = model.find_last()[1]

    inference_model.load_weights(model_path, by_name=True)

    class_names = ['BG', 'building']  # In our case, we have 1 class for the background, and 1 class for building

    file_names = next(os.walk(IMAGE_DIR))[2]
    random_image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

    predictions = inference_model.detect([random_image] * config.BATCH_SIZE,
                                         verbose=1)  # We are replicating the same image to fill up the batch_size

    # Run on single example
    if (examples == "one"):
        p = predictions[0]
        visualize.display_instances(random_image, p['rois'], p['masks'], p['class_ids'], class_names, p['scores'])
    # Run on entire test set
    else:
        # Gather all JPG files in the test set as small batches
        files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        ALL_FILES = []
        _buffer = []
        for _idx, _file in enumerate(files):
            if len(_buffer) == config.IMAGES_PER_GPU * config.GPU_COUNT:
                ALL_FILES.append(_buffer)
                _buffer = []
            else:
                _buffer.append(_file)

        if len(_buffer) > 0:
            ALL_FILES.append(_buffer)

        # Iterate over all the batches and predict
        _final_object = []
        for files in tqdm.tqdm(ALL_FILES):
            images = [skimage.io.imread(x) for x in files]
            predictions = inference_model.detect(images, verbose=0)
            for _idx, r in enumerate(predictions):
                _file = files[_idx]
                image_id = int(_file.split("/")[-1].replace(".jpg", ""))
                for _idx, class_id in enumerate(r["class_ids"]):
                    if class_id == 1:
                        mask = r["masks"].astype(np.uint8)[:, :, _idx]
                        bbox = np.around(r["rois"][_idx], 1)
                        bbox = [float(x) for x in bbox]
                        _result = {}
                        _result["image_id"] = image_id
                        _result["category_id"] = 100
                        _result["score"] = float(r["scores"][_idx])
                        _mask = maskUtils.encode(np.asfortranarray(mask))
                        _mask["counts"] = _mask["counts"].decode("UTF-8")
                        _result["segmentation"] = _mask
                        _result["bbox"] = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]
                        _final_object.append(_result)

def CompilePredictions():
    fp = open("predictions.json", "w")
    import json
    print("Writing JSON...")
    fp.write(json.dumps(_final_object))
    fp.close()

if __name__ == '__main__':
    # Uncomment in order to train
    # train()

    # Trains using pre-trained weights
    test()