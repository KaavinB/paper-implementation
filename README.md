# AlexNet Implementation in PyTorch

This repository contains a PyTorch implementation of AlexNet, a groundbreaking convolutional neural network architecture introduced by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. This implementation is designed for image classification tasks, particularly on the ImageNet dataset.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)

## Overview

AlexNet was a pivotal model in the field of computer vision, significantly outperforming previous approaches on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. This implementation aims to replicate the original architecture and training process as closely as possible using modern deep learning frameworks.

## Requirements

- Python 3.6+
- PyTorch 1.0+
- torchvision
- CUDA-capable GPU (recommended for training)

You can install the required packages using:

```
pip install torch torchvision
```

## Project Structure

- `models.py`: Contains the AlexNet model implementation.
- `train.py`: Script for training the model on ImageNet data.
- `README.md`: This file, containing project documentation.

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/kaavinb/alexnet.git
   cd alexnet
   ```

2. Prepare your ImageNet dataset (or a subset like ImageNet-mini) and update the `TRAIN_DIR` and `VAL_DIR` paths in `train.py`.

3. Run the training script:
   ```
   python train.py
   ```

## Model Architecture

The implemented AlexNet architecture consists of:
- 5 convolutional layers
- 3 max pooling layers
- 2 local response normalization layers
- 3 fully connected layers
- ReLU activations
- Dropout for regularization

The model is designed to take 227x227 RGB images as input and output probabilities for 1000 classes.

## Training

The training process includes:
- Data augmentation (random crops, horizontal flips)
- Stochastic Gradient Descent (SGD) optimizer
- Learning rate decay
- Weight decay for regularization

You can modify hyperparameters such as learning rate, batch size, and number of epochs in the `train.py` file.

## References

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
2. ImageNet. http://www.image-net.org/

## Contributing

Contributions to improve the implementation or documentation are welcome. Please feel free to submit a pull request or open an issue.
