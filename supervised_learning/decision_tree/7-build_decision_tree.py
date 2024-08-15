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

    def pred(self, x):
        """
        Recursively predict the value for a given input x
        by traversing the tree based on the feature thresholds.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def pred(self, x):
        """
        Predict the value for a given input x.
        Since this is a leaf node, it returns its stored value.
        """
        return self.value


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

    def pred(self, x):
        """
        Start the prediction process from the root node for a given input x.
        """
        return self.root.pred(x)

    def update_predict(self):
        """
        Updates the tree's prediction function to use
        vectorized operations for faster predictions.
        It calculates the indicator functions for each
        leaf and uses them to make predictions.
        """
        self.update_bounds()  # Compute bounds for each node in the tree
        leaves = self.get_leaves()  # Retrieve all the leaves in the tree

        for leaf in leaves:
            # Compute the indicator function for each leaf
            leaf.update_indicator()
        # Create a vectorized prediction function using
        # NumPy's vectorized operations
        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves], axis=0)

    def fit_node(self, node):
        """
        Recursively fits the tree starting from the given node.

        Args:
            node (Node): The node from which to start fitting the tree.

        This method splits the node if the conditions permit, or converts
        it into a leaf if the split conditions are not met (based on depth,
        population, or purity).
        """
        # Determine the feature and threshold for splitting this node
        node.feature, node.threshold = self.split_criterion(node)

        # Determine the sub-populations for the left and right child nodes
        left_population = (
          node.sub_population
          & (self.explanatory[:, node.feature] > node.threshold)
          )
        right_population = node.sub_population & ~left_population

        # Check if the left child should be a leaf
        is_left_leaf = (
            # Reached maximum depth
            node.depth == self.max_depth - 1 or
            # Too few individuals
            np.sum(left_population) <= self.min_pop or
            # All individuals belong to the same class
            np.unique(self.target[left_population]).size == 1
        )

        # Create the left child node
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Check if the right child should be a leaf
        is_right_leaf = (
            # Reached maximum depth
            node.depth == self.max_depth - 1 or
            # Too few individuals
            np.sum(right_population) <= self.min_pop or
            # All individuals belong to the same class
            np.unique(self.target[right_population]).size == 1
        )

        # Create the right child node
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Creates a leaf node for the given sub-population
        Args:
            node (Node): The current node
            sub_population (np.ndarray): The boolean mask
            of the sub-population for this leaf
        Returns:
            Leaf: A new Leaf node.
        """
        # Most common target value in the sub-population
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates a non-leaf (internal) node for further splitting
        Args:
            node (Node): The current node
            sub_population (np.ndarray): The boolean mask of
            the sub-population for this node

        Returns:
            Node: A new Node.
        """
        new_node = Node()
        new_node.depth = node.depth + 1
        new_node.sub_population = sub_population
        return new_node

    def accuracy(self, test_explanatory, test_target):
        """
        Calculates the accuracy of the decision tree on the test dataset
        Args:
            test_explanatory (np.ndarray): The feature matrix
            of the test dataset
            test_target (np.ndarray): The target values of
            the test dataset
        Returns:
            float: The accuracy of the model
        """
        predictions = self.predict(test_explanatory)
        return np.mean(predictions == test_target)

    def fit(self, explanatory, target, verbose=0):
        """
        Fits the decision tree to the given explanatory and target data.

        Args:
            explanatory (np.ndarray): The feature matrix
            of shape (n_individuals, n_features)
            target (np.ndarray): The target values
            of shape (n_individuals,)
            verbose (int): If set to 1, prints training
            summary after fitting
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            # Assuming this will be implemented later
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        # Recursively fit the tree starting from the root node
        self.fit_node(self.root)

        # Update the prediction function after fitting the tree
        self.update_predict()

        if verbose == 1:
            print(
                f"""  Training finished.
            - Depth                     : {self.depth()}
            - Number of nodes           : {self.count_nodes()}
            - Number of leaves          : {self.count_nodes(only_leaves=True)}
            - Accuracy on training data : """
                f"{self.accuracy(self.explanatory, self.target)}"""
            )

    def extreme(self, arr):
        """
        Returns the minimum and maximum values of a numpy array.
        Args:
            arr (np.ndarray): Input array.
        Returns:
            tuple: (min_value, max_value) of the array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Randomly selects a feature and threshold
        for splitting the node
        Args:
            node (Node): The current node for
            which the split is being determined
        Returns:
            tuple: A tuple (feature, threshold)
            where 'feature' is the index of the
            feature to split on, and 'threshold' is the value to split at.
        """
        diff = 0
        while diff == 0:
            # Randomly choose a feature index
            feature = self.rng.integers(0, self.explanatory.shape[1])
            # Find the minimum and maximum values of
            # this feature among the individuals
            feature_min, feature_max = (
              self.extreme(self.explanatory[:, feature][node.sub_population])
              )
            # Calculate the difference to ensure a valid threshold
            diff = feature_max - feature_min
        # Randomly choose a threshold within the range of the feature values
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold
