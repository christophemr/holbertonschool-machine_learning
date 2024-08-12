# Decision Tree & Random Forest

## Overview

This project focuses on implementing decision trees and random forests from scratch. The goal is to develop a deep understanding of how these machine learning models are constructed and used for prediction tasks. We will start by building the decision tree model, and gradually expand our work to include random forests and isolation forests, which are used for outlier detection.

### Concepts

For this project, it's important to review and understand the following concepts:

- **What is a decision tree?**
- **How does DecisionTree.pred differ from DecisionTree.predict?**
- **How are continuous attributes handled in decision trees?**
- **What is the Gini index and how is it used in splitting?**
- **What are Random Forests and how do they differ from decision trees?**
- **How can isolation forests be used for outlier detection?**

### Resources

To gain a deeper understanding of these concepts, the following resources are recommended:

1. **Rokach and Maimon (2002):** Top-down induction of decision trees classifiers: A survey.
2. **Ho et al. (1995):** Random Decision Forests.
3. **Fei et al. (2008):** Isolation Forests.
4. **Gini and Entropy clearly explained:** Handling Continuous Features in Decision Trees.
5. **Abspoel et al. (2021):** Secure Training of Decision Trees with Continuous Attributes.
6. **Threshold Split Selection Algorithm for Continuous Features in Decision Trees.**
7. **Splitting Continuous Attribute using Gini Index in Decision Tree.**
8. **How to handle Continuous Valued Attributes in Decision Trees.**
9. **Decision Tree problem based on the Continuous-valued Attribute.**
10. **How to Implement Decision Trees in Python using Scikit-Learn (sklearn).**
11. **Matching and Prediction on the Principle of Biological Classification by William A. Belson.**

### Notes

This project aims to implement decision trees from scratch. It is important for engineers to understand how the tools we use are built for two reasons:

1. It gives us confidence in our skills.
2. It helps us when we need to build our own tools to solve unsolved problems.

The first three references point to historical papers where the concepts were first studied. References 4 to 9 can help if you need further explanations about how to split nodes. William A. Belson is usually credited for the invention of decision trees (read reference 11). Despite our efforts to make it efficient, we cannot compete with Sklearn’s implementations (since they are done in C). In real life, it is recommended to use Sklearn’s tools.

### Requirements

- Carefully read all the concept pages attached above.
- All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.9).
- Your files will be executed with numpy (version 1.25.2).
- All your files should end with a new line.
- Your code should use the pycodestyle style (version 2.11.1).
- The first line of all your files should be exactly `#!/usr/bin/env python3`.
- A `README.md` file, at the root of the folder of the project, is mandatory.
- All your modules should have documentation.
- All your classes should have documentation.
- All your functions (inside and outside a class) should have documentation.
- All your files must be executable.

### Tasks

1. **Depth of a decision tree**  
   Implement a method to find the maximum depth of a decision tree.  
   **File:** `0-build_decision_tree.py`

2. **Number of nodes/leaves in a decision tree**  
   Count the number of nodes and leaves in a decision tree.  
   **File:** `1-build_decision_tree.py`

3. **Let's print our Tree**  
   Implement a method to print the structure of the decision tree.  
   **File:** `2-build_decision_tree.py`

4. **Towards the predict function (1): the get_leaves method**  
   Implement a method to get all the leaves of the decision tree.  
   **File:** `3-build_decision_tree.py`

5. **Towards the predict function (2): the update_bounds method**  
   Compute and store the bounds of each node in the decision tree.  
   **File:** `4-build_decision_tree.py`

6. **The predict function**  
   Implement the prediction function for the decision tree.  
   **File:** `5-build_decision_tree.py`

7. **Training decision trees**  
   Implement the fit method to train the decision tree on a dataset.  
   **File:** `7-build_decision_tree.py`

8. **Using Gini impurity function as a splitting criterion**  
   Implement Gini index-based splitting in the decision tree.  
   **File:** `8-build_decision_tree.py`

9. **Random forests**  
   Implement a random forest class that builds a collection of decision trees.  
   **File:** `9-random_forest.py`

10. **IRF 1: isolation random trees**  
    Implement isolation trees for outlier detection.  
    **File:** `10-isolation_tree.py`

11. **IRF 2: isolation random forests**  
    Implement an isolation forest for outlier detection.  
    **File:** `11-isolation_forest.py`
