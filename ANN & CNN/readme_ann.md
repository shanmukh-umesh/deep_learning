# Artificial Neural Network (ANN) for Classification

This project demonstrates how to build and train an Artificial Neural Network (ANN) using TensorFlow and Keras for a binary classification task.

---

## 🧠 Objective

The objective of this project is to classify customer churn based on input features using a neural network model.

---

## 📝 Project Structure

- `ANN_dl.ipynb`: Main Jupyter Notebook containing the end-to-end process:
  - Importing and preprocessing data
  - Building an ANN model using Keras Sequential API
  - Training and evaluating the model
  - Visualizing model performance

---

## ⚙️ Requirements

To run this notebook, ensure you have the following dependencies installed:

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow (2.x)
- Jupyter Notebook

## 🏗️ Model Architecture
Model Type: Sequential ANN (Feedforward)

Input Layer: Based on the number of input features

Hidden Layers: 2 Dense layers with ReLU activation

Output Layer: 1 Dense node with Sigmoid activation (for binary classification)

Loss Function: Binary Crossentropy

Optimizer: Adam

Metrics: Accuracy

## ✅ Output
The notebook provides:

Training vs. validation accuracy and loss graphs

Final model accuracy on test data

Optional: Confusion matrix or classification report (if implemented)
