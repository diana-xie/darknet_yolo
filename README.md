# Yolo-v4 and Yolo-v3 for deep-weeding-bot project

Improving object detection using YOLO v4/v3 for [deep-weeding-bot](https://github.com/diana-xie/deep-weeding-bot) project. Object detection is customized for corn vs. weed detection, in actual footage from robot traversing corn fields.

<img src="https://github.com/diana-xie/deep-weeding-bot/blob/master/documentation/weed-object-detection.gif?raw=true" width="500px">

- Libraries used: 
  * YOLO v4: [YOLO v4](https://github.com/AlexeyAB/darknet)
  * Labelling images: [labelImg](https://github.com/tzutalin/labelImg)
- Tutorials used:
  * [How to train YOLOv4 for custom objects detection in Google Colab](https://medium.com/ai-world/how-to-train-yolov4-for-custom-objects-detection-in-google-colab-1e934b8ef685)

# Results: 
- Notebook: [yolo_v4.ipynb](https://github.com/diana-xie/darknet_yolo/blob/master/yolo_v4.ipynb)
- Data augmentation code: [yolo_setup](https://github.com/diana-xie/deep-weeding-bot/tree/master/yolo_setup/yolo_v4)

# Overview

## Goal: 

Create an object detection algorithm that distinguishes corn from weeds, on real-life video data collected in the field.

<b>1. Distinguish corn from weeds</b>
* as long as weeds can be distinguished from "other" entities the majority of time

<b>2. Detect weeds with high recall</b>
* robot will only target weeds for action, therefore it is more critical to produce high recall; i.e. correctly predict positives (i.e. detect "weed") out of all the actual positives in the dataset

<img src="https://raw.githubusercontent.com/diana-xie/deep-weeding-bot/master/documentation/sample-detection.jpg" width="650px">

## Use case:
Use case is to simulate a scenario in which a robot is deployed in a corn field to kill the majority of weeds, while minimizing tradeoff of damage to corn crops. Thus, main objective is to "play it safe" and mainly target weeds with high recall, rather than produce a response to corn detection. 

## Approach: 
Train YOLO model on custom data. Custom data is actual video footage of robot traversing corn field, with weeds in path. The objects trained on are corn and weed labels in the footage, separated into images.

## Data:
- Video from robot with mounted GoPro camera, traversing actual corn fields with weeds growing on paths. 
- Manually labelled corn & weeds in footage (i.e. images extracted from footage), using [labelImg](https://github.com/tzutalin/labelImg)

<img src="https://raw.githubusercontent.com/diana-xie/deep-weeding-bot/master/documentation/labelImg.JPG" width="650px">

- Data augmentation to distort images and expand dataset

<img src="https://raw.githubusercontent.com/diana-xie/deep-weeding-bot/master/documentation/sample-augmentation.jpg" width="300px">
