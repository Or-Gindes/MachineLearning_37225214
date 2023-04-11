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
    """Evaluate using divorce dataset"""
    file_path = os.path.join(BASE_PATH, "divorce")
    divorce_data = np.genfromtxt(file_path, delimiter=';', skip_header=1)
    # Preprocess divorce_data

    # evaluate(data)