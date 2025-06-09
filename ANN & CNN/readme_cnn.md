## README: Image Classification with Keras

This repository contains a Jupyter Notebook (`CNN_dl.ipynb`) that demonstrates an image classification model using TensorFlow Keras. The model is trained on the CIFAR10 dataset.

### Overview

The `CNN_dl.ipynb` notebook provides a step-by-step implementation of an image classification model. It covers data loading, preprocessing, model definition, compilation, and training. While the notebook's title suggests a Convolutional Neural Network (CNN), the provided code snippets show an Artificial Neural Network (ANN) architecture with dense layers for image classification.

### Dataset

The model utilizes the CIFAR10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The 10 different classes represent airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

### Prerequisites

To run this notebook, you will need the following libraries installed:

* `tensorflow`
* `numpy`
* `matplotlib`


Run Cells Sequentially:
Execute the cells in the notebook sequentially.

### Import Libraries: The first cell imports necessary libraries from tensorflow.keras, numpy, and matplotlib.pyplot.
### Load Data: The CIFAR10 dataset is loaded using datasets.cifar10.load_data().
### Data Preprocessing:
* The y_train labels are reshaped into a 1D array.
* Image pixel values are normalized by dividing by 255 to scale them between 0 and 1.
* Class names for the CIFAR10 dataset are defined.
* A plot_image function is provided to visualize images from the dataset along with their corresponding labels.
* Model Definition: An ANN model is defined using tf.keras.Sequential(). It includes a Flatten layer to convert 32x32x3 images into a 1D array, followed by two Dense layers with relu activation and a final Dense layer with sigmoid activation for 10 output classes.
* Model Compilation: The model is compiled using the Stochastic Gradient Descent (SGD) optimizer, sparse_categorical_crossentropy as the loss function, and accuracy as the evaluation metric.
* Model Training: The model is trained on the preprocessed training data for 5 epochs.
