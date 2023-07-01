# CLIP_Hand_Demo

<p align="middle"> 
<img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/illustration.png" width="80%">
</p> 

## Introduction
A simple Zero-Shot demo for CLIP-Hand 3D.

<p align="middle"> 
<img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/matrix.png" width="90%">
</p> 

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
  
  + batch size = 16, id = 5, Zero-Shot Result: x: 99.99%, y: 99.99%, z: 99.99%
    <p align="middle"> 
    <img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/bs16.png" width="79%">
    </p> 
  + batch size = 32, id = 30, Zero-Shot Result: x: 99.99%, y: 99.99%, z: 85.93%
    <p align="middle"> 
    <img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/bs32.png" width="79%">
    </p> 
  + batch size = 64, id = 10, Zero-Shot Result: x: 99.99%, y: 99.99%, z: 73.95%
    <p align="middle"> 
    <img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/bs64.png" width="79%">
    </p> 
  + batch size = 128, id = 11, Zero-Shot Result: x: 91.54%, y: 99.98%, z: 86.88%
    <p align="middle"> 
    <img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/bs128.png" width="79%">
    </p> 
  + batch size = 256, id = 133, Zero-Shot Result: x: 88.28%, y: 93.92%, z: 88.22%
    <p align="middle"> 
    <img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/bs256_4.png" width="79%">
    </p> 

  
  
