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

        # recursively calculate the max depth from left & right children
        left_depth = (self.left_child.max_depth_below() if
                      self.left_child else self.depth)
        right_depth = (self.right_child.max_depth_below() if
                       self.right_child else self.depth)
        # returns the maximum depth found
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes or leaves
        in the subtree rooted at this node.
        Parameters:
            only_leaves (bool): If True, count only the leaf nodes.
        Returns:
            int: The total count of nodes or leaves.
        """
        if self.is_leaf:
            # returns 1 if the node is a leaf & only_leaves is true else 0
            return 1 if only_leaves else 1
        else:
            # Otherwise, continue counting in the left and right subtrees
            left_count = (self.left_child.count_nodes_below
                          (only_leaves) if self.left_child else 0)
            right_count = (self.right_child.count_nodes_below
                           (only_leaves) if self.right_child else 0)
            # For a non-leaf node, return the sum of the counts
            return ((left_count + right_count) if
                    only_leaves else (1 + left_count + right_count))

    def __str__(self):
        """
        Generates a string representation of the node and its subtree
        This method recursively creates a string that visually represents
        the node and its children, using prefixes to illustrate
        the tree structure
        Returns:
            str: The formatted string representing the node and its subtree
        """
        # Determine if the node is the root or an internal node
        current = "root" if self.is_root else "-> node"
        # Format the node information
        result = \
            f"{current} [feature={self.feature}, threshold={self.threshold}]\n"
        # Add left child with the correct prefix
        if self.left_child:
            result +=\
              self.left_child_add_prefix(str(self.left_child).strip())
        if self.right_child:
            # Add right child with the correct prefix
            result += \
              self.right_child_add_prefix(str(self.right_child).strip())
        return result

    def left_child_add_prefix(self, text):
        """
        Adds a prefix for the text of the left subtree.
        Parameters:
            text (str): The text to be prefixed.
        Returns:
            str: The text with the left child prefix added.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds a prefix for the text of the right subtree.
        Parameters:
            text (str): The text to be prefixed.
        Returns:
            str: The text with the right child prefix added.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def get_leaves_below(self):
        """Recursively retrieves all leaf nodes beneath this node
            in the decision tree
        Returns:
            list: A list of all leaf nodes found beneath this node
        """
        if self.is_leaf:
            return [self]

        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively updates bounds for this node and its children.
        Initializes at root with infinite bounds and adjusts for children
        based on data.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}
        for child in [self.left_child, self.right_child]:
            if child:
                # Make a copy of the current node's bounds to each child.
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()

                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                elif child == self.right_child:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """
        Update the indicator function for the node.
        The indicator function returns a boolean
        array where each element is True
        if the corresponding individual's features meet
        the conditions specified by
        the node's lower and upper bounds.
        """
        def is_large_enough(x):
            """
            Check if each individual's features are greater
            than the lower bounds.
            Args:
                x (np.ndarray): The input data array
                (n_individuals, n_features).
            Returns:
                np.ndarray: A boolean array indicating
                whether each individual meets
                the lower bounds criteria.
            """
            return (np.all([x[:, key] > self.lower[key]
                            for key in self.lower], axis=0))

        def is_small_enough(x):
            """
            Check if each individual's features are less than
            or equal to the upper bounds.
            Args:
                x (np.ndarray): The input data array
                (n_individuals, n_features).
            Returns:
                np.ndarray: A boolean array indicating
                whether each individual meets
                the upper bounds criteria.
            """
            return (np.all([x[:, key] <= self.upper[key]
                            for key in self.upper], axis=0))

        self.indicator = (lambda x:
                          np.logical_and(is_large_enough(x),
                                         is_small_enough(x)))


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

    def __str__(self):
        """returns a string representation of the node
        Returns:
            str: The formatted string representing the leaf node
        """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """
        Retrieves the leaf node if the current node is a leaf.
        Returns:
            list: A list containing the current node if it is a leaf.
    """
        return [self]

    def update_bounds_below(self):
        """
        Placeholder method for updating bounds in a leaf node.
        This method is intended to override the `update_bounds_below`
        method in the `Node` class.
        Since a leaf node does not have children and
        does not need to update any bounds,
        this method is implemented as a no-op (no operation).
        Returns:
            None
        """
        pass


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
        """Initializes the decision tree with parameters for tree construction
        and random number generation."""
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

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes or leaves in the decision tree.
        Parameters:
            only_leaves (bool): If True, count only the leaf nodes.
        Returns:
            int: The total number of nodes or leaves in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """returns a string representation of the entire decision tree
        Returns:
            str: The formatted string representing the decision tree
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Retrieves all leaf nodes in the decision tree.
        This method starts the process from the root of the tree and
        returns a list of all leaf nodes present in the entire tree.
        Returns:
            list: A list containing all the leaf nodes in the decision tree.
      """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Updates the bounds for all nodes in the decision tree"""
        self.root.update_bounds_below()
