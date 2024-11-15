# Dimensionality Reduction

## Description
This project focuses on implementing and understanding dimensionality reduction techniques such as Principal Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE). These methods reduce the number of dimensions in a dataset while retaining essential information, aiding in visualization, noise reduction, and computational efficiency.

---

## Resources
### Read or Watch:
- [Dimensionality Reduction For Dummies — Part 1: Intuition](#)
- [Singular Value Decomposition (SVD)](#)
- [Understanding SVD](#)
- [Dimensionality Reduction: Principal Components Analysis (PCA)](#)
- [StatQuest: t-SNE, Clearly Explained](#)
- [How to Use t-SNE Effectively](#)

### Definitions to Skim:
- **Dimensionality Reduction**
- **Principal Component Analysis**
- **Singular Value Decomposition**
- **Manifold**
- **Kullback–Leibler Divergence**
- **t-SNE**

---

## Learning Objectives
By the end of this project, you will be able to:
1. Explain **eigendecomposition** and **singular value decomposition (SVD)**.
2. Distinguish between **PCA** and **t-SNE** and understand their purposes.
3. Identify linear vs. non-linear dimensionality reduction techniques.
4. Implement PCA and t-SNE in Python to transform datasets effectively.

---

## Requirements
- **Language**: Python 3.9
- **Libraries**: `numpy` (version 1.25.2)
- **OS**: Ubuntu 20.04 LTS
- **Style**: Conform to `pycodestyle` (version 2.11.1)
- **Execution**:
  - All Python files must start with `#!/usr/bin/env python3`.
  - Code should minimize floating-point errors.

---

## Tasks

### 0. PCA
**Objective**: Implement PCA to reduce a dataset's dimensionality while maintaining a specified fraction of variance.

#### File: `0-pca.py`
- **Function**: `def pca(X, var=0.95):`
  - Input: 
    - `X`: Dataset of shape (n, d), centered with zero mean.
    - `var`: Fraction of variance to retain.
  - Output: Weights matrix \( W \) of shape (d, nd).
- **Method**:
  - Compute the **SVD** of \( X \).
  - Use singular values to calculate cumulative variance.
  - Select the top components to retain the desired variance.

**Example**:
```bash
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./0-main.py
[[-16.71379391   3.25277063  -3.21956297]
 ...
 [ 15.87543091   0.3804009    3.67627751]
 [  7.38044431  -1.58972122   0.60154138]]
1.7353180054998176e-29

### 1. PCA v2
**Objective**: Perform PCA to transform the dataset into a specified number of dimensions.

#### File: `1-pca.py`
- **Function**: def pca(X, ndim):
  - Input:
  - vX: Dataset of shape (n, d).
  - ndim: New dimensionality of 𝑋.
Output: Transformed dataset 𝑇 T of shape (n, ndim).

alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./1-main.py
X: (2500, 784)
T: (2500, 50)
[[-0.61344587  1.37452188 -1.41781926 ...  0.02276617  0.1076424 ]]
