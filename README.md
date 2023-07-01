# CLIP_Hand_Demo

## Introduction
A simple Zero-Shot demo for CLIP-Hand 3D.

## Install
+ Environment
  ```
  conda create -n cliphand python=3.9
  conda activate cliphand
  ```
+ Package Requirements
  ```
  pip install -r requirements.txt
  ```
+ Install CLIP
  Please follow [CLIP links](https://github.com/openai/CLIP) to install CLIP module.
+ Download Pre-train Weights
  [Google Driver](https://drive.google.com/file/d/1yergw0w1XtIkgh2feippylnVyTcF44Hm/view?usp=sharing)

## Download Datasets
FreiHAND Dataset
+ Download [link](https://lmb.informatik.uni-freiburg.de/projects/freihand/)

## Run Zero Shot Demo
+ Put Pre-train Weights on your Ubuntu Disk, and record its location.
  ```
  /home/anonymous/CLIP_HAND_3D_0402.pth.tar
  ```
+ Extract the FreiHAND dataset to Ubuntu Disk and record its location.
  ```
  /home/anonymous/FreiHAND
  ```
+ Modify Parameters in demo/main.py
  ```
  WEIGHT_PATH = YOUR_WEIGHT_PATH
  DATASET_PATH = YOUR_FreiHAND_PATH
  BATCH_SIZE = YOUR_BS
  ```
+ Run and Output
  ```
  python main.py
  ```
  + BS = 16, Id = 5, Zero-Shot Result.
    <p align="middle"> 
    <img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/bs16.png" width="75%">
    </p> 

  
  
