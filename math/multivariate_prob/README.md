# multivariate_prob
## Description
This project covers the basics of multivariate probability, including joint probability distributions, covariance, and correlation matrices. It also introduces the multivariate Gaussian distribution and its applications. The project involves implementing key concepts in Python, adhering to rigorous documentation and code style standards.

## Learning Objectives
By the end of this project, you should be able to:

Explain the contributions of Carl Friedrich Gauss.
Define joint and multivariate distributions.
Understand covariance and correlation coefficients.
Calculate and interpret covariance matrices.
Describe and implement the multivariate Gaussian distribution.

## Resources
### Required Reading
* Joint Probability Distributions
* Multivariate Gaussian Distributions
* The Multivariate Gaussian Distribution
* An Introduction to Variance, Covariance & Correlation
* Variance-Covariance Matrix Using Matrix Notation of Factor Analysis

### Definitions to Skim
* Carl Friedrich Gauss
* Joint Probability Distribution
* Covariance
* Covariance Matrix
### Reference Material
* numpy.cov
* numpy.corrcoef
* numpy.linalg.det
* numpy.linalg.inv
* numpy.random.multivariate_normal

## Requirements
* Python version: 3.9
* NumPy version: 1.25.2
* All code must conform to the pycodestyle standard (version 2.11.1).
* Documentation is mandatory for:
    * Modules
    * Classes
    * Functions
* Code must be executable and end with a new line.
* You are not allowed to import libraries beyond import numpy as np.

## Mandatory Tasks:
0. [Mean and Covariance](/math/multivariate_prob/0-mean_cov.py)
Objective: Implement a function to calculate the mean and covariance matrix of a dataset.

1. [Correlation](/math/multivariate_prob/1-correlation.py)
Objective: Implement a function to calculate the correlation matrix from a covariance matrix.
2. [Initialize](/math/multivariate_prob/multinormal.py)
Objective: Create a class MultiNormal to represent a multivariate normal distribution.
3. [PDF](/math/multivariate_prob/multinormal.py)
Objective: Add a method to MultiNormal to calculate the probability density function (PDF) for a given data point.
