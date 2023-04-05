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
    if len(y) == 0:
        return 0
    p1 = np.count_nonzero(y == 1) / y.size
    entropy = -p1 * math.log2(p1) - (1 - p1) * math.log2(1 - p1) if (p1 != 0 and p1 != 1) else 0
    return entropy


