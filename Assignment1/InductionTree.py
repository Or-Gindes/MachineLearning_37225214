"""
Assignment1 - Implementation of a decision tree algorithm called ID3 for binary features and preformance evaluation
Authors: Or Gindes & Alexandra Chilikov
"""
import math
import numpy as np


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
