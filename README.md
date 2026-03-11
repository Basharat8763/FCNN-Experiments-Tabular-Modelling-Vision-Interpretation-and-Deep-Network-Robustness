# Fully Connected Neural Networks (FCNN) Experiments  
Tabular Modelling, Vision Learning, and Deep Network Robustness

## Overview

This project explores the behavior, capabilities, and limitations of **Fully Connected Neural Networks (FCNNs)** across different types of machine learning tasks. The experiments investigate how FCNN architectures perform on:

1. Tabular Data
2. Image Classification
3. Deep Network Training Stability

The work demonstrates important deep learning concepts such as:

- Feature scaling in tabular neural networks
- Feature learning in vision models
- Sensitivity of FCNNs to spatial structure
- Vanishing gradient problem in deep networks
- Impact of activation functions, normalization, and optimizers

Three experimental studies were conducted using:

- **UCI Adult Census Income Dataset**
- **MNIST Handwritten Digit Dataset**
- **Tiny ImageNet Subset**

The models were implemented using **PyTorch** and trained on a **CUDA-enabled GPU**.

---

# Repository Structure

The repository is organized to make experiments reproducible and easy to understand.

```
fcnn-deep-learning-experiments
│
├── notebooks
│   ├── part1_tabular_modelling.ipynb
│   ├── part2_mnist_analysis.ipynb
│   └── part3_deep_network_experiments.ipynb
│
├── datasets
│   ├── adult
│   │   └── adult dataset files
│   │
│   ├── mnist
│   │   └── MNIST csv dataset
│   │
│   └── tiny-imagenet
│       └── README.txt (dataset download instructions)
│
├── figures
│   ├── weight_heatmaps
│   ├── gradient_norm_plots
│
├── results
│   ├── experiment_logs
│   └── accuracy_tables
│
├── reports
│   └── fcnn_assignment_report.pdf
│
├── requirements.txt
│
└── README.md
```

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# Dataset Information

## 1. UCI Adult Census Income Dataset

Used for **tabular classification**.

The dataset contains demographic attributes of individuals and predicts whether their income exceeds \$50K per year.

Features include:

Numerical features

- age
- fnlwgt
- education-num
- capital-gain
- capital-loss
- hours-per-week

Categorical features

- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

Target variable:

- ≤50K → Class 0
- >50K → Class 1

Download it from:

https://archive.ics.uci.edu/dataset/2/adult

---

## 2. MNIST Handwritten Digit Dataset

Used for **image classification and feature interpretation**.

Dataset properties:

- 60,000 training images
- 10,000 test images
- image size: 28 × 28 pixels
- grayscale

Each image is flattened into a **784-dimensional vector** for FCNN training.

Due to GitHub size limitations, the dataset is **not included in this repository**.

Download it from:

https://www.kaggle.com/datasets/hojjatk/mnist-dataset

After downloading, extract the dataset inside:

```
datasets/MNIST/
```


---

## 3. Tiny ImageNet Subset

Used for **deep network stress testing and robustness experiments**.

Dataset properties:

- 10 object classes
- image size: 64 × 64 RGB
- images flattened into a **12,288 dimensional input vector**

Due to GitHub size limitations, the dataset is **not included in this repository**.

Download it from:

https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet

After downloading, extract the dataset inside:

```
datasets/tiny-imagenet/
```

---

# Part 1 — Tabular Modelling (Adult Census Dataset)

## Objective

Build a **Fully Connected Neural Network** to model the relationship between demographic attributes and income level.

The experiment evaluates the impact of **feature scaling** on neural network training.

---

## Model Architecture

```
Input Layer → 128 → 64 → Output (1)
```

Training configuration

- Activation Function: ReLU
- Loss Function: BCEWithLogitsLoss
- Optimizer: Adam
- Batch Size: 256
- Epochs: 20
- Device: CUDA GPU

---

## Experiments

Two experiments were conducted.

### Experiment 1 — Raw Numerical Features

The model was trained without feature normalization.

### Experiment 2 — Min-Max Scaled Features

Numerical features were normalized into the range `[0,1]`.

---

## Results

| Experiment | Test Accuracy |
|------------|--------------|
| Raw Data | 78.68% |
| Scaled Data | 84.35% |

Feature scaling improved model accuracy by approximately **5.7%** and significantly stabilized training.

---

# Part 2 — Vision Learning and Feature Interpretation (MNIST)

## Objective

Analyze how a Fully Connected Neural Network learns visual representations and whether spatial structure is important.

Two analyses were performed:

1. **Weight visualization of the first hidden layer**
2. **Pixel shuffle experiment**

---

## Model Architecture

```
784 → 256 → 128 → 10
```

Training configuration

- Activation: ReLU
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 15
- Batch Size: 256

---

## Standard MNIST Results

Final Test Accuracy

**97.94%**

Despite lacking spatial awareness, FCNNs successfully learn discriminative representations for digit classification.

---

## Weight Visualization

The first hidden layer contains **256 neurons**, each receiving **784 input weights**.

These weights were reshaped into **28×28 matrices** and visualized as heatmaps.

Observations include:

- circular patterns corresponding to digits like **0 and 8**
- vertical stroke patterns for **digit 1**
- diagonal patterns for **digits like 4 or 7**

These results indicate that FCNNs learn primitive visual detectors.

---

## Pixel Shuffle Experiment

To test spatial sensitivity:

- a fixed random permutation was applied to pixel indices
- the same permutation was used for all images

Results:

| Dataset | Accuracy |
|-------|---------|
| Standard MNIST | 97.94% |
| Shuffled MNIST | 97.75% |

The negligible difference demonstrates that **FCNNs do not rely on spatial structure**.

---

# Part 3 — Deep Network Stress Testing (Tiny ImageNet)

## Objective

Evaluate training stability of **very deep fully connected networks** and experimentally demonstrate the **vanishing gradient problem**.

---

## Deep FCNN Architecture

```
12288 → 2048 → 1024 → 1024 → 512 → 512 → 256 → 128 → 64 → 10
```

Two configurations were tested:

1. Sigmoid Activation
2. ReLU + Batch Normalization + Dropout

---

## Vanishing Gradient Experiment

| Model | Validation Accuracy |
|------|--------------------|
| Deep Sigmoid Network | 16.8% |
| ReLU + BatchNorm | 41.4% |

Sigmoid networks experienced severe gradient decay:

```
Gradient Norm: 1e-6 → 1e-11
```

ReLU with Batch Normalization maintained healthy gradient flow:

```
Gradient Norm: 30 → 0.2 – 0.5
```

---

# Ablation Study

Controlled modifications were applied to the ReLU + BatchNorm baseline model.

| Configuration | Validation Accuracy |
|--------------|--------------------|
| Baseline (Adam, lr=0.001) | 41.4% |
| No Dropout | 41.2% |
| Learning Rate ×10 | 41.2% |
| Learning Rate ÷10 | 34.6% |
| SGD Optimizer | 13.0% |
| Deep Sigmoid Network | 16.8% |

---

# Key Findings

- Feature scaling significantly improves FCNN performance on tabular data.
- FCNNs can learn meaningful visual patterns even without spatial modeling.
- Pixel shuffling has minimal effect on FCNN performance.
- Deep sigmoid networks suffer from severe vanishing gradients.
- ReLU and Batch Normalization significantly improve training stability.
- Optimizer choice strongly influences deep network performance.

---

# Running the Experiments

1. Clone the repository

```
git clone https://github.com/yourusername/fcnn-deep-learning-experiments.git
cd fcnn-deep-learning-experiments
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run experiments using the Jupyter notebooks located in the `notebooks/` directory.

---

# Results and Visualizations

Visual outputs such as:

- weight heatmaps
- gradient norm plots
- training curves

are stored in the `figures/` directory.

Experiment outputs and logs are stored in the `results/` directory.

---

# Author

Basharat

Focus Area: Artificial Intelligence, Machine Learning, and Data Science
