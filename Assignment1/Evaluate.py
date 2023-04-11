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
    y = divorce_data[:, 1]
    est = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform').fit(X)
    X_transformed = est.transform(X)
    results[data_name] = evaluate(X_transformed, y)
    pass