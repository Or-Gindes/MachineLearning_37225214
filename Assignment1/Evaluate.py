"""
Assignment1 - Implementation of a decision tree algorithm called ID3 for binary features and preformance evaluation
Authors: Or Gindes & Alexandra Chilikov
"""
import os
import numpy as np
from InductionTree import evaluate
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, LabelBinarizer

BASE_PATH = "C:\\Users\\Or\\PycharmProjects\\MachineLearning_37225214\\Assignment1\\datasets"

if __name__ == "__main__":
    results = dict()
    """Evaluate using divorce dataset - https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set"""
    data_name = "divorce"
    file_path = os.path.join(BASE_PATH, f"{data_name}.csv")
    divorce_data = np.genfromtxt(file_path, delimiter=';', skip_header=1)
    # Preprocess divorce_data
    np.random.shuffle(divorce_data)
    X = divorce_data[:, :-1]
    y = divorce_data[:, -1]
    # Each feature in X is a question response between 0 and 4
    # We'll discretize the answers into two bins, 0 = [0,1] and 1 = [2,3,4]
    est = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform').fit(X)
    X_transformed = est.transform(X)
    results[data_name] = evaluate(X_transformed, y, repetitions=3, data_name=data_name, sync=True)
    pass
    # TODO: repeat this process with 4 more classification datasets
    # TODO: write detailed report -
    #  Evaluation report: Write a short report presenting the detailed results and summarizing your findings,
    #  including which model performed the best, and why.
    #  the report should include a table that compares the predictive performance of the various methods,
    #  in the following structure: Dataset | Method | Evaluation metric | Evaluation Value | Fit Runtime (in ms)
    """Evaluate using dataset #2 - link """

    """Evaluate using dataset #3 - link """

    """Evaluate using dataset #4 - link """

    """Evaluate using dataset #5 - link """
