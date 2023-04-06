"""
Assignment1 - Implementation of a decision tree algorithm called ID3 for binary features and preformance evaluation
Authors: Or Gindes & Alexandra Chilikov
"""
import math
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def compute_entropy(y):
    """
    Computes the entropy for array y

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           positive (`1`) or negative (`0`)

    Returns:
        entropy (float): Entropy at that node
    """
    # Verify y isn't empty and return entropy = 0 if it is
    if len(y) == 0:
        return 0
    p1 = np.count_nonzero(y == 1) / y.size
    # if p1 is zero or one, assign entropy = 0 to avoid ValueError of log(0)
    entropy = -p1 * math.log2(p1) - (1 - p1) * math.log2(1 - p1) if (p1 != 0 and p1 != 1) else 0
    return entropy


def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        node_indices (list):    List containing the active indices. I.e, the samples being considered at this step.
        feature (int):          Index of feature to split on

    Returns:
        left_indices (list):    Indices with feature value == 1
        right_indices (list):   Indices with feature value == 0
    """

    # You need to return the following variables correctly
    left_indices = np.array(node_indices)[np.where(X[node_indices, feature] == 1)[0]].tolist()
    right_indices = np.array(node_indices)[np.where(X[node_indices, feature] == 0)[0]].tolist()
    return left_indices, right_indices


def compute_information_gain(X, y, node_indices, feature):
    """
    Compute the information of splitting the node on a given feature

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        feature (int):          Index of feature to split on

    Returns:
        cost (float):        Cost computed
    """
    left, right = split_dataset(X, node_indices, feature)
    node_entropy = compute_entropy(y[node_indices])
    left_entropy = compute_entropy(y[left])
    right_entropy = compute_entropy(y[right])
    weighted_entropy = (len(left) / len(node_indices) * left_entropy + len(right) / len(node_indices) * right_entropy)
    cost = node_entropy - weighted_entropy
    return cost


def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature and threshold value
    to split the node data

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """
    gains = [compute_information_gain(X, y, node_indices, feature) for feature in range(X.shape[1])]
    best_feature = np.argmax(gains)
    return best_feature


class Node:
    """
    Support class for binary decision tree build of nodes
    """
    def __init__(self, pred_value=None, split_feature=None, left_child=None, right_child=None):
        """
        Args:
            pred_value (int): since the tree is binary this will be either zero or one.
            split_feature (int or None): index of feature to split by or None of this is a terminal node in the tree
            left_child (Node or None): Node to the left of this one or None of this is a terminal node in the tree
            right_child (Node or None): Node to the right of this one or None of this is a terminal node in the tree
        """
        self.pred_value = pred_value
        self.split_feature = split_feature
        self.left_child = left_child
        self.right_child = right_child


class MyID3(BaseEstimator, ClassifierMixin):
    """
    Decision tree base estimator class - MyID3
    BaseEstimator provides get_params and set_params methods
    ClassifierMixin provides a score method
    successively picking the best feature to split on until we reach the maximum depth defined by the user
    """
    def __init__(self, max_depth=None):
        """
        Initializes the decision tree estimator with give max_depth parameters
        Args:
            max_depth (int or None):    maximum depth of the decision tree or None for unlimited depth
        """
        self._max_depth = max_depth
        self._n_features= None
        self._tree = None

    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set (X, y)
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
        """
        # Check that X and y have correct shape
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have mismatching shapes")

        self._n_features = X.shape[1]
        self._tree = self._build_tree(X, y)

        return self

    def _build_tree(self, X, y, depth=0):
        """
        Recursive function which implements the ID3 algorithm and returns the root node of the tree
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable

        Returns:
            root_node (Node):       tree root_node which connects to the remaining tree nodes
        """
        # stopping criteria -
        # 1. All samples in node have the same target value
        if len(np.unique(y)) == 1:
            return Node(pred_value=y[0])
        # 2. No more features to split by
        # TODO:
        # 3. only one smaple in node
        if len(y) == 1:
            return Node(pred_value=y[0])
        # 4. Maximum depth
        if self._max_depth and depth >= self._max_depth:
            return Node(pred_value=int(np.argmax(np.unique(y, return_counts=True)[1])))

        # no stopping criteria matched
        best_feature = get_best_split()# TODO:


        return root_node

    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X.
        The predicted class probability is the fraction of samples of the same class in a leaf.
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)

        Returns:
            Prob (ndarray)                of shape (n_samples, n_classes)
        """
        if self._tree is None:
            raise AssertionError("Model hasn't been fitted yet")

    def predict(self, X):
        """
        Predict class with the highest probability in the function predict_proba
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)

        Returns:
            y (array like) of shape (n_samples,)
        """
        if self._tree is None:
            raise AssertionError("Model hasn't been fitted yet")


