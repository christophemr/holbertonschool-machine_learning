#!/usr/bin/env python3
"""
This module implements a Random Forest classifier from scratch.
It includes methods for training the forest, making predictions,
and evaluating the model's accuracy.
"""

import numpy as np

Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    """
    A Random Forest classifier composed of multiple decision trees.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the Random Forest model.

        Args:
            n_trees (int): Number of trees in the forest. Default is 100.
            max_depth (int): Maximum depth of each tree. Default is 10.
            min_pop (int): Minimum population required to split a node.
                           Default is 1.
            seed (int): Random seed for reproducibility. Default is 0.
        """
        # List to store prediction functions for each tree
        self.numpy_preds = []
        self.target = None  # Stores the target values of the training data
        self.explanatory = None  # Stores the explanatory variables
        self.n_trees = n_trees  # Number of trees in the forest
        self.max_depth = max_depth  # Maximum depth allowed for each tree
        self.min_pop = min_pop  # Minimum population required to split a node
        self.seed = seed  # Seed for random number generation

    def predict(self, explanatory):
        """
        Predicts the class for each example in the input data
        using the Random Forest.

        Args:
            explanatory (ndarray): The input data for prediction,
                                   shape (n_samples, n_features).

        Returns:
            ndarray: The predicted classes, shape (n_samples,).
        """
        # Collect predictions from each tree in the forest
        tree_preds = np.array([
            predictor(explanatory) for predictor in self.numpy_preds
        ])

        # Calculate the mode (most frequent prediction) for each example
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds
        )

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        Trains the Random Forest model on the provided dataset.

        Args:
            explanatory (ndarray): Input features for the training data,
                                   shape (n_samples, n_features).
            target (ndarray): Target values for the training data,
                              shape (n_samples,).
            n_trees (int, optional): Number of trees to build in the forest.
                                     Default is 100.
            verbose (int, optional): If set to 1, prints out training info.
                                     Default is 0.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []  # Reset the list of prediction functions

        # Initialize lists to keep track of tree statistics
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        # Train each tree in the forest
        for i in range(n_trees):
            T = Decision_Tree(
                max_depth=self.max_depth, min_pop=self.min_pop,
                seed=self.seed + i
            )
            T.fit(explanatory, target)

            # Store the tree's prediction function
            self.numpy_preds.append(T.predict)

            # Collect statistics from the tree
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(explanatory, target))

        # If verbose mode is enabled, print out the training statistics
        if verbose == 1:
            print("  Training finished.")
            print(f"    - Mean depth                     : "
                  f"{np.array(depths).mean()}")
            print(f"    - Mean number of nodes           : "
                  f"{np.array(nodes).mean()}")
            print(f"    - Mean number of leaves          : "
                  f"{np.array(leaves).mean()}")
            print(f"    - Mean accuracy on training data : "
                  f"{np.array(accuracies).mean()}")
            print(f"    - Accuracy of the forest on td   : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def accuracy(self, test_explanatory, test_target):
        """
        Computes the accuracy of the Random Forest model on the test data.

        Args:
            test_explanatory (ndarray): Input features for the test data,
                                        shape (n_samples, n_features).
            test_target (ndarray): True target values for the test data,
                                   shape (n_samples,).

        Returns:
            float: The accuracy of the predictions compared to the true
                   target values.
        """
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
