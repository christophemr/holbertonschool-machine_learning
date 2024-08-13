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
        Returns a string representation of the node and its children.
        This method recursively generates the string representation of the current
        node and its children, applying the appropriate prefix to visually represent
        the tree structure.
        Returns:
            str: The string representation of the node and its subtree
        """
        # represents the current node with feature & threshold
        str_node = f"node [feature={self.feature}, threshold={self.threshold}]"
        # add the left child representation with correct prefix
        if self.left_child:
          str_node += "\n" + self.left_child_add_prefix(str(self.left_child))
        # add the right child representation with correct prefix
        if self.right_child:
          str_node += "\n" + self.right_child_add_prefix(str(self.right_child))
        return str_node

    def left_child_add_prefix(self, text):
        """
        Adds a prefix for the text of the left subtree.

        Parameters:
            text (str): The text to be prefixed.

        Returns:
            str: The text with the left child prefix added.
        """
        lines = text.split("\n")
        new_text = "    +--->" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |     " + line + "\n"
        return new_text.strip()

    def right_child_add_prefix(self, text):
        """
        Adds a prefix for the text of the right subtree.

        Parameters:
            text (str): The text to be prefixed.

        Returns:
            str: The text with the right child prefix added.
        """
        lines = text.split("\n")
        new_text = "    `--->" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |     " + line + "\n"
        return new_text.strip()


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
        """returns a string representation of the node"""
        return (f"-> leaf [value={self.value}]")


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
        """returns a string representation of the node"""
        return "root " + self.root.__str__()
