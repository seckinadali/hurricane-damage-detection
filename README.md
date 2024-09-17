# Hurricane Damage Detection in Satellite Imagery

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seckinadali/hurricane-damage-detection/blob/main/hurricane_damage_detection.ipynb)

This project focuses on automating the detection of buildings damaged by hurricanes using satellite imagery. The dataset used in this project consists of images captured after Hurricane Harvey in 2017. The primary goal is to build a binary classifier that can accurately distinguish between damaged and undamaged buildings in post-hurricane satellite imagery.

## Project Objective

The key objective is to develop a model that can:
- **Classify** buildings as either "damaged" or "undamaged" based on satellite images.
- **Achieve high accuracy** (> 90%) on a test set of over 10,000 images, despite training on a limited dataset of fewer than 1,000 images.

## Dataset

The dataset has been taken from [IEEE Dataport](https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized) and modified to feature a small training set of fewer than 1,000 images and a large test set of more than 10,000 images. This modification simulates the challenges of training on limited data while requiring high performance on a larger, more diverse test set.

## Approach

Given the challenges of working with a small dataset, this project involves building and experimenting with various Convolutional Neural Network (CNN) models and training techniques to optimize performance.

### Experiments Conducted

1. **Baseline CNN Model**: 
   - A simple CNN architecture with 2-3 layers.
   - Achieved 83% accuracy.
   - This model serves as a baseline for further improvements.

2. **Tweaked CNN with Data Augmentation and Batch Normalization**:
   - Applied random rotation and contrast adjustments to augment the data.
   - Incorporated batch normalization to stabilize and accelerate training.
   - Achieved 92% accuracy, a significant improvement over the baseline model.

3. **EfficientNet B0 Models**:
   - Experimented with EfficientNet B0, a more complex and pre-trained architecture.
   - **Version 1**: All layers frozen, resulting in 91% accuracy.
   - **Version 2**: Unfroze the last block of layers to fine-tune the model, resulting in 86% accuracy.
   - **Version 3**: Unfroze the last two blocks of layers, also achieving 86% accuracy.

### Training Techniques

- **Early Stopping**: To prevent overfitting by stopping training when performance on a validation set starts to degrade.
- **Exponential Decay Learning Rate Scheduler**: Adjusted the learning rate during training to optimize convergence.

### Evaluation

- **Metrics Used**: Precision, recall, and accuracy were the primary metrics for evaluating model performance.
- **Confusion Matrix**: Used to visualize the performance of the models on the test data.
- **Accuracy Plots**: Plotted training and validation accuracy together to better understand the model's learning process.

## Results

- **Test Scores**:
  - Baseline CNN: **83%**
  - Tweaked CNN: **92%**
  - EfficientNet B0 (all layers frozen): **91%**
  - EfficientNet B0 (last block unfrozen): **86%**
  - EfficientNet B0 (last two blocks unfrozen): **86%**
