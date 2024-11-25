# USpeech Project

## Team Contributions

### **Gaurav Sharma**  
#### **Preprocessing**
- Implemented and managed the data preprocessing scripts to prepare datasets for training and testing.
- Key Scripts:
  - `LRW_dataset_preprocessing.py`: Handles preprocessing of the Lip Reading in the Wild (LRW) dataset, including video and audio synchronization and cleaning.
  - `collected_dataset_preprocessing.py`: Processes collected datasets, ensuring format uniformity and preparing inputs for further stages.
  - `noisy_collected_dataset_preprocessing.py`: Focused on augmenting datasets with noise for robust model training.

### **Yash Diggikar**  
#### **USpeech Enhancement Model**
- Developed and trained the speech enhancement model, focusing on improving audio quality in ultrasound-based speech enhancement.
- Key Scripts:
  - `model.py`: Defines the deep learning architecture used for speech enhancement, leveraging techniques like UNet.
  - `loss.py`: Implements custom loss functions to optimize the enhancement model's performance.
  - `train.py`: Manages the training pipeline, including data loading, optimization, and validation.
  - `modules/UNet.py`: Contains the UNet model definition for speech enhancement tasks.
  - `modules/bottleneck_transformer.py`: Integrates transformer modules for feature bottleneck enhancement.

#### **Speech Synthesis Model** *(Initial Work Started)*  
- Began foundational work on the speech synthesis model for ultrasound-based speech generation.
- Key Scripts:
  - `uspeech_synthesis_model/modules/ultrasoundAudioModel.py`: Initial design for integrating ultrasound input with audio synthesis.
  - `uspeech_synthesis_model/pretrain_dataset.py`: Script for handling pretraining datasets for synthesis tasks.

---

This document outlines the contributions of team members and provides an overview of the technical work completed for the USpeech project. Updates will be added as progress continues.



