"""
Assignment1 - Implementation of a decision tree algorithm called ID3 for binary features and preformance evaluation
Authors: Or Gindes & Alexandra Chilikov
"""
import os
import numpy as np
from InductionTree import evaluate
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, LabelBinarizer
import pandas as pd

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
    # results[data_name] = evaluate(X_transformed, y, repetitions=3, data_name=data_name, sync=False)
    pass
    # TODO: repeat this process with 4 more classification datasets
    # TODO: write detailed report -
    #  Evaluation report: Write a short report presenting the detailed results and summarizing your findings,
    #  including which model performed the best, and why.
    #  the report should include a table that compares the predictive performance of the various methods,
    #  in the following structure: Dataset | Method | Evaluation metric | Evaluation Value | Fit Runtime (in ms)
    """Evaluate using dataset #2 - link """
    df = pd.read_csv('C:\\Users\\Or\\PycharmProjects\\MachineLearning_37225214\\Assignment1\\datasets\\diabetes_prediction_dataset.csv')
    X = df.drop(['diabetes'], axis=1)
    y = df['diabetes']
    encoder = OneHotEncoder(sparse=False)
    cols_to_encode = ['gender', 'smoking_history']
    encoded_cols = encoder.fit_transform(X.loc[:, cols_to_encode])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names(cols_to_encode))
    X = pd.concat([X.drop(columns=cols_to_encode), encoded_df], axis=1)

    # Discretize
    discretize_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
    X.loc[:, discretize_cols] = discretizer.fit_transform(X.loc[:, discretize_cols])
    data_name = "diabetes.csv"
    results[data_name] = evaluate(X, y, data_name, repetitions=2, n_folds=5, sync=False)
    """Evaluate using dataset #3 - link """
    pass
    """Evaluate using dataset #4 - link """

    """Evaluate using dataset #5 - link """
