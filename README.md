# CLIP_Hand_Demo

:loudspeaker: **Update (10/09/2023):** Our paper "CLIP-Hand3D: Exploiting 3D Hand Pose Estimation via Context-Aware Prompting" has been accepted at **ACM MM 2023**! Stay tuned for more updates. :tada:

<p align="middle"> 
<img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/illustration.png" width="80%"> 
</p> 

CLIP-Hand3D: Exploiting 3D Hand Pose Estimation via Context-Aware Prompting

*Shaoxiang Guo, Qing Cai*, Lin Qi and Junyu Dong* (*Corresponding Authors)*

*School of Computer Science and Technology, Ocean University of China, 238 Songling Road, Qingdao, China.* 

## Introduction
In our paper, we introduce CLIP-Hand3D, a novel method for 3D hand pose estimation from monocular images using Contrastive Language-Image Pre-training (CLIP). We bridge the gap between text prompts and the irregular distribution of hand joint positions in 3D space by encoding pose labels into text representations and hand joint spatial distribution into pose-aware features. We maximize the semantic consistency between pose-text features using a CLIP-based contrastive learning paradigm. Our method, which includes a coarse-to-fine mesh regressor, achieves comparable SOTA performance and significantly faster inference speed on several public hand benchmarks. 
In this github repository, we will release the corresponding codes. First, we release a simple zero-shot demo to show the semantic relations between hand images and pose text prompts.

<p align="middle"> 
<img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/pipeline.png" width="85%"> 
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
    <img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/bs16.png" width="85%">
    </p> 
  + batch size = 32, id = 30, Zero-Shot Result: x: 99.99%, y: 99.99%, z: 85.93%
    <p align="middle"> 
    <img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/bs32.png" width="85%">
    </p> 
  + batch size = 64, id = 10, Zero-Shot Result: x: 99.99%, y: 99.99%, z: 73.95%
    <p align="middle"> 
    <img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/bs64.png" width="85%">
    </p> 
  + batch size = 128, id = 11, Zero-Shot Result: x: 91.54%, y: 99.98%, z: 86.88%
    <p align="middle"> 
    <img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/bs128.png" width="85%">
    </p> 
  + batch size = 256, id = 133, Zero-Shot Result: x: 88.28%, y: 93.92%, z: 88.22%
    <p align="middle"> 
    <img src="https://github.com/ShaoXiang23/CLIP_Hand_Demo/blob/main/images/bs256_4.png" width="79%">
    </p> 
  
  
