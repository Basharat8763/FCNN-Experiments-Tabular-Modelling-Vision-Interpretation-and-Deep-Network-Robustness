# FCNN-Experiments-Tabular-Modelling-Vision-Interpretation-and-Deep-Network-Robustness

## Overview
This project explores the design, training, and analysis of Fully Connected Neural Networks (FCNNs) across different data modalities and experimental settings. The work investigates how FCNN architectures behave when applied to tabular datasets, image data, and deep network training scenarios.

The project is divided into three major experimental sections:

1. Tabular data modelling using the UCI Adult Census Income dataset
2. Vision-based analysis using the MNIST handwritten digit dataset
3. Stress testing and robustness experiments on Tiny ImageNet

The objective is to understand both the capabilities and limitations of fully connected neural networks in practical machine learning tasks.

---

## Project Structure

The project is organized into three main experimental components:
Part 1 – Tabular Modelling (UCI Adult Dataset)
Part 2 – Vision & Feature Interpretation (MNIST)
Part 3 – Stress Testing & Robustness (Tiny ImageNet)


Each section contains code for dataset processing, model construction, training, evaluation, and experiment analysis.

---

## Part 1: Tabular Modelling (UCI Adult Census Income Dataset)

### Objective
Model the relationship between demographic features and income level using a fully connected neural network.

### Dataset
UCI Adult Census Income Dataset

The dataset contains demographic attributes such as:
- Age
- Workclass
- Education
- Occupation
- Capital Gain
- Hours per week
- Marital status
- Native country

The target variable predicts whether an individual's income exceeds \$50K per year.

### Model Architecture
A 3-layer Fully Connected Neural Network was implemented using a deep learning framework.

Typical architecture:
Input Layer
Hidden Layer 1
Hidden Layer 2
Output Layer (Binary Classification)



### Data Processing
The following preprocessing steps were applied:

- Encoding categorical variables
- Handling missing values
- Feature normalization
- Min-Max scaling for numerical attributes

### Experiments
Two training experiments were conducted:

1. Model trained on raw feature values
2. Model trained on Min-Max scaled data

### Result
The scaled model demonstrated faster convergence and more stable optimization compared to the raw feature model.

---

## Part 2: Vision and Feature Interpretation (MNIST Dataset)

### Objective
Investigate how fully connected neural networks interpret pixel-based image data.

### Dataset
MNIST Handwritten Digits Dataset

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 grayscale

Each image contains a handwritten digit between 0 and 9.

### Model
A Fully Connected Neural Network was trained on flattened image vectors.
Input: 784 neurons (28 × 28 pixels)
Hidden Layers
Output Layer: 10 classes



### Weight Visualization
To understand how the network learns visual patterns:

1. The weights from the first hidden layer were extracted.
2. Each neuron in the hidden layer contains 784 incoming weights.
3. These weights were reshaped into a 28 × 28 matrix.
4. The matrices were visualized using heatmaps.

### Interpretation
The visualizations show that certain neurons become sensitive to specific pixel regions or patterns such as curves or edges that are characteristic of particular digits.

---

### Flattening Experiment

To test whether the FCNN relies on spatial structure:

1. The pixels of every MNIST image were randomly shuffled.
2. The same shuffle pattern was applied to all images.
3. The model was trained on this scrambled dataset.

### Observation

The FCNN achieved similar performance on both standard and shuffled datasets.

### Explanation

Fully connected neural networks treat inputs as independent features and do not preserve spatial relationships between pixels. As a result, the network can still learn statistical patterns even when the spatial arrangement is disrupted.

This experiment highlights why convolutional neural networks (CNNs) are more suitable for image tasks, as they explicitly capture spatial locality and structure.

---

## Part 3: Stress Testing and Robustness (Tiny ImageNet)

### Objective
Evaluate training stability and robustness in deeper fully connected networks.

### Dataset
Tiny ImageNet (subset)

The dataset contains small resolution images across multiple classes and was used to test deeper FCNN architectures.

---

### Vanishing Gradient Experiment

A very deep FCNN (8+ layers) was trained under two configurations.

#### Experiment A
Activation Function: Sigmoid

This configuration exhibited slow convergence due to the vanishing gradient problem, where gradients diminish as they propagate through deeper layers.

#### Experiment B
Activation Function: ReLU with Batch Normalization

ReLU activations and batch normalization improved gradient flow and significantly accelerated training.

Gradient norms for the first layer were monitored to compare the training dynamics between the two experiments.

---

### Ablation Study

To analyze the effect of different training components, several controlled modifications were applied:

1. Dropout removed
2. Learning rate increased and decreased by a factor of 10
3. Optimizer switched from Adam to vanilla SGD

These experiments helped identify which training components had the largest influence on model performance.

---

## Technologies Used

- Python
- PyTorch / TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Key Learnings

This project demonstrates several important concepts in deep learning:

- Importance of feature scaling for neural network training
- Interpretability through weight visualization
- Limitations of fully connected networks in image tasks
- Vanishing gradient problems in deep architectures
- Training stability improvements using ReLU and batch normalization
- Impact of optimizer choice and hyperparameters

---

## Reproducibility

All experiments were implemented using standard deep learning frameworks and publicly available datasets. The repository includes scripts for:

- Data preprocessing
- Model training
- Experiment replication
- Visualization of learned features

---

