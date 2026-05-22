# Privacy Benchmarking

A modular benchmark suite for evaluating privacy risks, privacy-preserving machine learning methods, and privacy attacks across generative and discriminative models.

---

## Overview

This repository aims to provide a standardized framework for:

- Evaluating privacy leakage in machine learning models
- Benchmarking privacy attacks and defenses
- Comparing privacy-preserving training methods
- Reproducing published privacy research results
- Providing consistent datasets, metrics, and evaluation pipelines

## Initial TODO:
1. Generate a large set of CelebA synthetic images using StyleGAN3 (10,000 imgs)
2. Train various types of downstream models (Resnet18) to classify CelebA images (on real train data, on synthetic data, on real train data using DP-SGD)
3. Implement a white-box and black-box attack. Measure attack efficacy on various downstream models. Measure accuracy, precision, recall of downstream models on test data.

## References: 

https://arxiv.org/abs/2604.05256 Protecting and Preserving Protest Dynamics for Responsible Analysis

https://arxiv.org/abs/1812.00910 White-Box Attack

## Hypothesis: 
Models trained using synthetic data will have better trade-offs of privacy-utility than those on real train data and those using DP-SGD.

## Control Variables: 
Hyper-parameter settings, epochs, architectures for attack and downstream models, 
