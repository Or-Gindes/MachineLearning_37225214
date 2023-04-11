"""
Assignment1 - Implementation of a decision tree algorithm called ID3 for binary features and preformance evaluation
Authors: Or Gindes & Alexandra Chilikov
"""
import math
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
import wandb


# 1. Calculate entropy
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


# 2. Split dataset
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


# 3. Calculate information gain
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


# 4. Get best split
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


# 5. Building the tree
class Node:
    """
    Support class for binary decision tree build of nodes
    """
    def __init__(self, node_probas=None, split_feature=None, left_child=None, right_child=None):
        """
        Args:
            node_probas (array like or None): since the tree is binary this will be list for terminal nodes
                                        with fraction of samples of the same class or None for other nodes.
            split_feature (int or None): index of feature to split by or None of this is a terminal node in the tree
            left_child (Node or None): Node to the left of this one or None of this is a terminal node in the tree
            right_child (Node or None): Node to the right of this one or None of this is a terminal node in the tree
        """
        self.node_probas = node_probas
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
        self.max_depth = max_depth
        self._n_features = None
        self._tree = None
        self._n_classes = None

    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set (X, y)
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
        """
        # Check that X and y have correct shape
        if len(X) != len(y):
            raise ValueError("X and y have mismatching shapes")

        self._n_classes = len(np.unique(y)) # will be 2 because of binary tree
        self._n_features = X.shape[1]
        self._tree = self._build_tree(X, y, list(range(X.shape[0])))

    def _build_tree(self, X, y, node_indices, depth=0):
        """
        Recursive function which implements the ID3 algorithm and returns the root node of the tree
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

        Returns:
            root_node (Node):       tree root_node which connects to the remaining tree nodes
        """
        # stopping criteria -
        # 1 + 2. only one sample in node / All samples in node have the same target value - pure node
        if len(node_indices) == 1 or len(np.unique(y[node_indices])) == 1:
            pred_probas = np.zeros(self._n_classes)
            pred_probas[y[0]] = 1.0
            return Node(node_probas=pred_probas)
        # 3 + 4. No more features to split by / 4. Maximum depth - mixed node
        if depth == self._n_features or (self.max_depth and depth >= self.max_depth):
            return Node(node_probas=np.unique(y[node_indices], return_counts=True)[1] / len(node_indices))

        # no stopping criteria matched - find best feature for split
        best_feature = get_best_split(X, y, node_indices)

        split_node = Node(split_feature=best_feature)
        left, right = split_dataset(X, node_indices, best_feature)
        split_node.left_child = self._build_tree(X, y, node_indices=left, depth=depth + 1)
        split_node.right_child = self._build_tree(X, y, node_indices=right, depth=depth + 1)

        return split_node

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

        pred_probas = np.zeros((len(X), self._n_classes))
        for i, x in enumerate(X):
            node = self._tree
            while node.split_feature is not None:
                # left:    feature value == 1
                if x[node.split_feature] == 1:
                    node = node.left_child
                # right:   feature value == 0
                else:
                    node = node.right_child
            pred_probas[i] = node.node_probas
        return pred_probas

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

        pred_probas = self.predict_proba(X)
        return np.argmax(pred_probas, axis=1)


# 6. Bagging Algorithm
class MyBaggingID3(BaseEstimator, ClassifierMixin):
    """
    Ensemble model using MyID3 as base classifier
    BaseEstimator provides get_params and set_params methods
    ClassifierMixin provides a score method
    """
    def __init__(self, n_estimators=3, max_samples=0.5, max_features=0.5, max_depth=None):
        """
        Initializes the decision tree estimator with give max_depth parameters
        Args:
            n_estimators (int):         The number of ID3 in the ensemble
            max_samples (float):        float between 0 and 1 - represents the fraction of *samples* to draw from X
                                        *with replacement* to train each base estimator
            max_features (float):       float between 0 and 1 - represents the fraction of *features* to draw from X
                                        *without replacement* to train each base estimator
            max_depth (int or None):    maximum depth of the decision tree or None for unlimited depth
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_depth = max_depth
        self._estimators = []
        self._n_classes = None

    def fit(self, X, y):
        """
        Build an ensemble model from the training set (X, y), using MyID3 as base classifier
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
        """
        # Check that X and y have correct shape
        if len(X) != len(y):
            raise ValueError("X and y have mismatching shapes")

        self._n_classes = len(np.unique(y))
        # fit n base estimators
        for i in range(self.n_estimators):
            # randomly select training samples from X with replacement for each base classifier
            tree_indices = np.random.choice(range(X.shape[0]), size=math.ceil(self.max_samples*X.shape[0]), replace=True)

            # randomly select features from X without replacements from for each base classifier
            tree_features = np.random.choice(range(X.shape[1]), size=math.ceil(self.max_features*X.shape[1]), replace=False)

            base_tree = MyID3(max_depth=self.max_depth)
            base_tree.fit(X=X[np.ix_(tree_indices, tree_features)], y=y[tree_indices])

            self._estimators.append(base_tree)

    def predict_proba(self, X):
        """
        Predicts the class labels for X using the ensemble model.
        The predicted class probability is average of prediction probabilities from each base classifier.
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)

        Returns:
            Prob (ndarray)                of shape (n_samples, n_classes)
        """
        if len(self._estimators) == 0:
            raise AssertionError("Model hasn't been fitted yet")

        base_predictions = np.zeros((len(X), self.n_estimators))
        for i, base_tree in enumerate(self._estimators):
            base_predictions[:, i] = base_tree.predict(X)

        pred_probas = np.zeros((len(X), self._n_classes))
        for j, x in enumerate(X):
            probas = np.unique(base_predictions[j, :], return_counts=True)[1] / self.n_estimators
            if len(probas) == 1:
                pred_probas[j, int(base_predictions[j, :][0])] = 1.0
            else:
                pred_probas[j] = probas

        return pred_probas

    def predict(self, X):
        """
        Predict class with the highest probability in the function predict_proba
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)

        Returns:
            y (array like) of shape (n_samples,)
        """
        if len(self._estimators) == 0:
            raise AssertionError("Model hasn't been fitted yet")

        pred_probas = self.predict_proba(X)
        return np.argmax(pred_probas, axis=1)


# 7. Evaluation
def evaluate(binary_array, repetitions=2, n_folds=5):
    """
    Compares the performance of MyID3 and MyBaggingID3 models against sklearn DecisionTreeClassifier and
    BaggingClassifier respectively use repeated K-fold cross-validation to evaluate both algorithms,
    Using at least 2 repetitions and at least 5 folds and returns Evaluation metrics

    Args:
        binary_array (ndarray): Preprocessed binary Data matrix of shape(n_samples, n_features)
        repetitions (int): number of repetitions >= 2
        n_folds (int): number of folds >= 5

    Returns:
        evaluation_dict (dict): Dictionary of evaluation scores - Accuracy, Precision, Recall, F1-score, ROC_AUC score
        from sklearn - accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    """
    # TODO: Implement using weights & biases package to collect the results
    # tutorial - https://colab.research.google.com/github/wandb/examples/blob/master/colabs/scikit/Simple_Scikit_Integration.ipynb#scrollTo=PusIQpdPzUbP
    # create new project in https://wandb.ai/machinelearning_37225214/projects
    pass


if __name__ == "__main__":
    # Debugging
    X = np.random.randint(0, 2, size=(100, 3))
    y = np.random.randint(0, 2, size=100)
    X_test = [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0]]
    tree_bag = MyBaggingID3(n_estimators=3, max_depth=2)
    tree_bag.fit(X, y)
    # tree_bag.predict_proba(X_test)
    tree_bag.predict(X_test)
