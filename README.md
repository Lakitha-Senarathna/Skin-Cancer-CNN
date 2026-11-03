# Skin Cancer CNN

A deep learning project for automated skin cancer classification using Convolutional Neural Networks (CNNs). This repository contains a Jupyter Notebook workflow for preparing image data, training a CNN model, and evaluating classification performance on skin lesion images.

The project is designed around a medical image classification task, where the model learns to distinguish between skin lesion categories such as benign and malignant cases.

## Project Overview

Skin cancer is one of the most common types of cancer worldwide, and early detection can significantly improve treatment outcomes. This project explores the use of computer vision and deep learning to classify skin lesion images automatically.

The repository includes:
- A data preparation notebook to resize and normalize skin lesion images
- A training/evaluation notebook for building and testing a CNN
- A saved trained model in H5 format
- A notebook-based workflow suitable for Google Colab or Jupyter environments

## Repository Contents

- `ML_Final_Project.ipynb` – main notebook containing the data pipeline, model training, and evaluation workflow
- `ML_Project_Image_Resize.ipynb` – notebook used to resize images to a fixed dimension before model input
- `skin_cancer_best_model.h5` – trained Keras CNN model saved as a serialized HDF5 file

## Problem Statement

The goal of this project is to build a CNN-based classifier that can analyze skin lesion images and predict whether they are benign or malignant. This can assist clinicians by supporting early screening and diagnosis.

## Objectives

- Load and preprocess skin lesion images
- Resize images to a consistent input shape
- Train a CNN model using labeled image data
- Evaluate classification performance
- Save the best-performing model for reuse

## Dataset

The project uses a collection of labeled skin lesion images organized into class directories. The resize notebook indicates images were prepared into folders such as:

- `Benign`
- `Malignant`

The images are resized to a standard size, such as 224x224 pixels, before training.

Note: The repository does not include a full dataset description file in the root directory, so the exact dataset source should be confirmed before publication or distribution.

## Workflow

### 1. Image Preparation
The notebook `ML_Project_Image_Resize.ipynb` uses PIL (Python Imaging Library) to:
- read images from a source folder
- resize each image to a fixed dimension
- save the processed images to a prepared output directory

Example workflow:
- Input image directory: dataset folder
- Output image directory: resized dataset folder
- Resize operation: 224x224 pixels

### 2. Model Training
The main notebook (`ML_Final_Project.ipynb`) appears to:
- load resized images
- build a CNN using TensorFlow/Keras
- set up data generators or image arrays
- train the model on skin lesion classes
- monitor performance
- save the best model

### 3. Model Inference
The trained model can be loaded and used to predict the class of a new image. The saved model file is:

```bash
skin_cancer_best_model.h5
