# AlexNet Implementation in PyTorch

This repository contains an implementation of AlexNet using PyTorch, designed for image classification tasks.

## How to Run

1. Clone the repository, install dependencies, and run training:

   git clone https://github.com/kaavinb/alexnet.git && cd alexnet && pip install torch torchvision && python train.py

## Model Overview

- 5 convolutional layers with normalization and pooling
- Fully connected layers with Dropout and ReLU
- Softmax for classification

## References

- [AlexNet Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
