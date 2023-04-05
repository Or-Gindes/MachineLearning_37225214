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
    left_indices = []
    right_indices = []
