# BuildingNet

This project implements a baseline Mask R-CNN for segmenting buildings from satellite imagery.

### Mask R-CNN

The Mask R-CNN is a framework for object instance segmentation. It initially uses a Region Proposal Network with Non-max Suppression to identify bounding boxes for objects of interest. These are complemented by masks generated using an additional branch (composed of a Fully Convolutional Network) that determines whether or not a pixel is part of an object. The model takes as input 300x300 3-channel (RGB) images, annotated in the MS-COCO format. The implementation uses a ResNet 101 as the backbone.

### Sample Results

<img src="assets/pic1.png" alt="Picture1" width="400px" height="400px"></img>
<img src="assets/pic2.png" alt="Picture2" width="400px" height="400px"></img>

### Using the Model

To train the model, edit `Training-Predictions.py` by uncommenting the call to `train()` and follow the comment in order to test using the trained model. In order to test using pretrained weights, add the pretrained weights file according to the following file structure:

```
ROOT
├── mrcnn
├── data
|   ├── pretrained_weights.h5
│   ├── test
│   │   └── images/
│   │   └── annotation.json
│   ├── train
│   │   └── images/
│   │   └── annotation.json
│   └── val
│       └── images/
│       └── annotation.json
```
The `pretrained_weights.h5` can be downloaded from [https://www.crowdai.org/challenges/mapping-challenge/dataset_files](https://www.crowdai.org/challenges/mapping-challenge/dataset_files).

### Sources

- This code is heavily inspired by the implementation by Sharada Mohanty for the CrowdAI Mapping Challenge
(https://www.crowdai.org/challenges/mapping-challenge) with minor changes
- Adapted from the tensorflow implementation available at https://github.com/matterport/Mask_RCNN
- Mask R-CNN (https://arxiv.org/abs/1703.06870)
