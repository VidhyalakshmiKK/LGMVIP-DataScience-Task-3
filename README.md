# Handwritten Digit Recognition Using ANN and CNN
## 📋 Table of Contents
- Introduction
- Technologies Used
- Dataset
- Model Architectures
- Results
## 📖 Introduction
This project implements Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) to recognize handwritten digits from the famous **MNIST dataset**. CNN is especially suited for image-based tasks, while ANN provides a basic comparison for understanding the power of convolution layers in deep learning.

### Objective:
- To classify handwritten digits using both ANN and CNN models.
- To analyze the performance difference between these two architectures in solving image recognition problems.

## 🛠 Technologies Used
- Python: Core programming language.
- TensorFlow / Keras: For building and training the neural networks (ANN and CNN).
- Pandas: For data manipulation and preprocessing.
- NumPy: For numerical computations.
- Matplotlib / Seaborn: Data visualization to analyze the results.
## 📊 Dataset
The project uses the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits (0-9). Each image is grayscale, 28x28 pixels, representing digits from 0 to 9.

- Input shape: (28, 28, 1) for CNN, flattened to (784,) for ANN.
- Output: One-hot encoded labels representing digits (0-9).

## 🧠 Model Architectures
**1. Artificial Neural Network (ANN)**

The ANN used here is a simple feedforward neural network. It consists of fully connected layers:

- Input Layer: Flattened images (28x28 = 784 nodes).
- Hidden Layers: One or two fully connected layers with ReLU activation.
- Output Layer: 10 nodes (one for each digit), with softmax activation.
- 
**2. Convolutional Neural Network (CNN)**
CNN utilizes convolutional layers to extract features from the images:

- Input Layer: (28x28x1) grayscale images.
- Convolution Layers: Multiple Conv2D layers followed by max-pooling.
- Fully Connected Layer: Dense layer for classification.
- Output Layer: 10 nodes with softmax activation for multi-class classification.
## 📈 Results
### **Performance Comparison:**
**ANN Model Accuracy:** ~97% on test data.

Misclassified data from ANN model

![image](https://github.com/user-attachments/assets/dfa9225a-e4a4-4921-b9ea-ef57f794490c)

**CNN Model Accuracy:** ~98% on test data.

![image](https://github.com/user-attachments/assets/b4c8f0aa-62f7-42c7-95b2-7175fd26dfe2)

Accuracy score of misclassified images from ANN model has increased by 67% in the CNN model
