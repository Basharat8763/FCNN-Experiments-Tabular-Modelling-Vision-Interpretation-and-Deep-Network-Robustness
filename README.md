# Fully Connected Neural Networks (FCNN) for Tabular Data, Vision Tasks, and Deep Network Robustness

## Overview

This project explores the design, training behavior, and limitations of Fully Connected Neural Networks (FCNNs) across different types of machine learning tasks. The objective was to investigate how FCNN architectures perform on tabular data, image classification tasks, and deep network training scenarios under different optimization and architectural conditions.

The project is divided into three major experimental components:

1. Tabular Modelling using the UCI Adult Census Income dataset  
2. Vision Learning and Feature Interpretation using the MNIST dataset  
3. Stress Testing and Robustness Analysis using a Tiny ImageNet subset  

These experiments collectively demonstrate the importance of feature preprocessing, architectural design, activation functions, optimizer choice, and normalization techniques in neural network training.

---

# Part 1 — Tabular Modelling (UCI Adult Census Income Dataset)

## Objective

The objective of this task was to build a Fully Connected Neural Network (FCNN) to model the relationship between demographic attributes and income level using the UCI Adult Census dataset. The problem is formulated as a binary classification task where the model predicts whether an individual's income exceeds \$50K per year.

Target Classes:

- ≤50K → Class 0  
- >50K → Class 1  

The experiment also evaluates the impact of feature scaling on training stability and model performance.

---

## Dataset Description

The dataset contains demographic and socio-economic attributes of individuals.

### Numerical Features
- Age  
- fnlwgt  
- education-num  
- capital-gain  
- capital-loss  
- hours-per-week  

### Categorical Features
- workclass  
- education  
- marital-status  
- occupation  
- relationship  
- race  
- sex  
- native-country  

Missing values in the dataset appear as `"?"` and were treated as a valid category rather than removing rows to avoid data loss.

The target variable originally contained string labels (`<=50K`, `>50K`) which were converted into binary values.

---

## Data Processing Pipeline

To prepare the dataset for neural network training, the following preprocessing steps were applied:

1. **String Cleaning**  
   All categorical variables were converted to lowercase and stripped of extra whitespace.

2. **Handling Missing Values**  
   Missing entries represented by `"?"` were retained as separate categories.

3. **Label Encoding**  
   Income values were converted to binary labels.

4. **One-Hot Encoding**  
   All categorical features were transformed using one-hot encoding.

5. **Column Alignment**  
   Ensured training and test feature matrices had identical feature dimensions.

6. **Feature Scaling (Second Experiment Only)**  
   Min-Max scaling was applied to numerical features to normalize them into the range `[0,1]`.

---

## Model Architecture

A 3-layer Fully Connected Neural Network was implemented using PyTorch.

Architecture:

Input Layer → 128 neurons → 64 neurons → Output Layer (1 neuron)

Training Configuration:

- Activation Function: ReLU  
- Loss Function: BCEWithLogitsLoss  
- Optimizer: Adam  
- Batch Size: 256  
- Epochs: 20  
- Device: CUDA-enabled GPU  

---

## Experiments

Two experiments were conducted.

### Experiment 1 — Raw Features

The model was trained using original numerical values without scaling.

### Experiment 2 — Min-Max Scaled Features

Numerical features were normalized using MinMaxScaler before training.

---

## Results

| Experiment | Test Accuracy |
|------------|--------------|
| Raw Data | 78.68% |
| Scaled Data | 84.35% |

Feature scaling improved accuracy by approximately **5.7%**.

---

## Analysis

The experiment demonstrates that Fully Connected Neural Networks are sensitive to feature scale. Large magnitude differences between numerical features can destabilize gradient updates and lead to irregular convergence.

Applying Min-Max scaling:

- Stabilized gradient updates  
- Produced smoother loss curves  
- Improved generalization performance  
- Increased final test accuracy  

---

## Conclusion

This experiment highlights the importance of proper feature preprocessing when training neural networks on tabular data. While the FCNN achieved reasonable performance without scaling, normalization significantly improved both convergence stability and model accuracy.

---

# Part 2 — Vision and Feature Interpretation (MNIST)

## Objective

This section investigates how Fully Connected Neural Networks process image data and what internal feature representations they learn. The MNIST handwritten digit dataset was used as a benchmark for multi-class classification.

Two major analyses were performed:

1. Visualization of the first hidden layer weights  
2. A pixel shuffle experiment to evaluate spatial awareness

---

## Dataset Description

MNIST consists of grayscale images of handwritten digits.

- 60,000 training samples  
- 10,000 test samples  
- Image size: 28 × 28 pixels  

Each sample contains:

- 1 label  
- 784 pixel values  

---

## Preprocessing

The following preprocessing steps were applied:

- Separated labels from pixel values  
- Normalized pixel values from `[0,255]` to `[0,1]`  
- Converted data into PyTorch tensors  
- Used DataLoader with batch size 256 for efficient GPU training  

Images were flattened into vectors since FCNNs operate on fixed-length input vectors.

---

## Model Architecture

A 3-layer FCNN was used.

Input (784) → Hidden Layer (256) → Hidden Layer (128) → Output (10)

Training Setup:

- Activation: ReLU  
- Loss: CrossEntropyLoss  
- Optimizer: Adam  
- Learning Rate: 0.001  
- Batch Size: 256  
- Epochs: 15  

---

## Standard MNIST Results

Training loss decreased smoothly across epochs indicating stable convergence.

Final Test Accuracy:

**97.94%**

This demonstrates that FCNNs can achieve strong performance even without explicit spatial modeling.

---

## Weight Visualization

The first hidden layer contains 256 neurons, each receiving 784 input weights.

For interpretation:

- The 784 weights of selected neurons were reshaped into **28×28 matrices**  
- These matrices were visualized using heatmaps  

Observations:

- Some neurons showed circular weight patterns, corresponding to digits like **0, 6, 8**  
- Some neurons highlighted vertical strokes associated with digit **1**  
- Other neurons detected diagonal patterns similar to digits **4 or 7**

This shows that FCNNs can learn primitive visual detectors in early layers.

---

## Pixel Shuffle Experiment

To test the importance of spatial structure:

- A fixed random permutation of the 784 pixel indices was generated  
- The same permutation was applied to every image in the dataset  
- The network was retrained on this shuffled dataset  

Results:

| Dataset | Test Accuracy |
|-------|---------------|
| Standard MNIST | 97.94% |
| Shuffled MNIST | 97.75% |

The performance difference was negligible.

---

## Analysis

This result reveals a key limitation of Fully Connected Neural Networks.

FCNNs treat input images as flat feature vectors and do not preserve spatial relationships between pixels. As long as the pixel permutation remains consistent across the dataset, the network can still learn statistical relationships between input positions and labels.

Humans rely heavily on spatial patterns to recognize digits, so shuffled images become difficult for humans to interpret. However, the FCNN continues to perform well because it does not rely on spatial structure.

---

## Architectural Insight

Unlike FCNNs, Convolutional Neural Networks (CNNs):

- Preserve spatial locality  
- Use local receptive fields  
- Employ weight sharing  
- Are translation invariant  

Therefore, CNNs are far more suitable for complex computer vision tasks.

---

# Part 3 — Stress Testing and Robustness (Tiny ImageNet)

## Objective

This section investigates the training stability of deep fully connected neural networks using a Tiny ImageNet subset.

Two main investigations were conducted:

1. Vanishing Gradient Demonstration  
2. Ablation Study of key training components  

---

## Dataset

Tiny ImageNet subset:

- 3,500 training images  
- 500 validation images  
- 10 object classes  
- Image resolution: 64×64 RGB  

Each image was flattened into a **12,288 dimensional vector** before entering the FCNN.

---

## Deep FCNN Architecture

A very deep fully connected architecture (8+ layers) was implemented.

12288 → 2048 → 1024 → 1024 → 512 → 512 → 256 → 128 → 64 → 10

Two configurations were tested.

### Experiment A — Sigmoid Activation

Validation Accuracy: **16.8%**

Gradient Norms:  
1e-6 → 1e-11

This indicates severe vanishing gradients, where early layers receive almost no updates.

---

### Experiment B — ReLU + Batch Normalisation

Validation Accuracy: **41.4%**

Gradient Norms:  
30 → 0.2–0.5

ReLU prevented saturation and Batch Normalization stabilized activations, enabling effective gradient flow.

---

## Gradient Analysis

The gradient norm plot clearly demonstrates the vanishing gradient problem.

Sigmoid networks showed collapsing gradients, while the ReLU + BatchNorm network maintained stable gradient magnitudes and trained successfully.

---

## Ablation Study

To evaluate robustness, controlled modifications were applied to the ReLU + BatchNorm baseline.

| Configuration | Validation Accuracy |
|---------------|--------------------|
| Baseline (Adam, lr=0.001) | 41.4% |
| No Dropout | 41.2% |
| Learning Rate ×10 | 41.2% |
| Learning Rate ÷10 | 34.6% |
| SGD Optimizer | 13.0% |
| Deep Sigmoid Network | 16.8% |

---

## Key Findings

1. Removing dropout had minimal impact on validation accuracy.  
2. Increasing learning rate by 10× did not destabilize training due to Adam's adaptive updates.  
3. Decreasing learning rate caused underfitting and slower convergence.  
4. Switching from Adam to SGD caused the largest performance drop.  
5. Deep sigmoid networks failed due to vanishing gradients.  
6. Activation function and optimizer choice had the greatest impact on model performance.

---

## Conclusion

This project experimentally demonstrates several important deep learning principles:

- The importance of feature scaling for tabular neural networks  
- The ability of FCNNs to learn visual features despite lacking spatial awareness  
- The insensitivity of FCNNs to spatial structure in images  
- The vanishing gradient problem in deep sigmoid networks  
- The effectiveness of ReLU and Batch Normalisation in deep architectures  
- The critical role of optimizer selection and learning rate tuning  

These results highlight both the capabilities and limitations of Fully Connected Neural Networks across different machine learning domains.

---

## Technologies Used

- Python  
- PyTorch  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## Author

B.Tech Computer Science and Engineering  
Focus Area: Artificial Intelligence, Machine Learning, and Data Science
