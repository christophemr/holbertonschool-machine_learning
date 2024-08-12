#!/usr/bin/env python3
"""
This module implements a simple decision tree classifier
from scratch.
It includes the Node and Leaf classes for constructing the tree,
as well as a Decision_Tree class for training and predicting
"""

import numpy as np


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        feature (int): The index of the feature used for
        splitting at this node.
        threshold (float): The threshold value for the feature at this node.
        left_child (Node or Leaf): The left child node
        resulting from the split.
        right_child (Node or Leaf): The right child node
        resulting from the split.
        is_leaf (bool): Indicates whether this node is a leaf.
        is_root (bool): Indicates whether this node is
        the root of the tree.
        sub_population (np.ndarray): The subset of data samples
        that reach this node.
        depth (int): The depth of this node in the tree.
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Computes the maximum depth of the subtree rooted at this node.

        Returns:
            int: The maximum depth from this node down to the deepest leaf.
        """
        # if the node is a leaf, return its depth
        if self.left_child is None and self.right_child is None:
            return self.depth

        # recursivelycalculate the max depth from left & right children
        left_depth = (self.left_child.max_depth_below() if
                      self.left_child else self.depth)
        right_depth = (self.right_child.max_depth_below() if
                       self.right_child else self.depth)
        # retunr the maximum depth found
        return max(left_depth, right_depth)


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.

    Attributes:
        value (any): The target value or class label that this leaf predicts.
        depth (int): The depth of this leaf in the tree.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of this leaf.

        Since a leaf does not have any children,
        its maximum depth is its own depth.

        Returns:
            int: The depth of this leaf node.
        """
        return self.depth


class Decision_Tree():
    """
    Implements a decision tree classifier.

    Attributes:
        max_depth (int): The maximum depth allowed for the tree.
        min_pop (int): The minimum number of samples required to split a node.
        seed (int): Seed for random number generator to ensure reproducibility.
        split_criterion (str): The criterion used to split nodes
        ('random', 'gini', etc.).
        root (Node): The root node of the decision tree.
        explanatory (np.ndarray): The feature matrix used for training.
        target (np.ndarray): The target values or class labels
        corresponding to the feature matrix.
        predict (callable): The prediction function once the tree is trained.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Calculates the maximum depth of the decision tree.

        Returns:
            int: The depth of the deepest node in the tree.
        """
        return self.root.max_depth_below()
