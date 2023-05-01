"""
Assignment1 - Implementation of a decision tree algorithm called ID3 for binary features and performance evaluation
Authors: Or Gindes & Hilla Halevi
"""
import os
import numpy as np
from InductionTree import MyID3, MyBaggingID3
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
import wandb

BASE_DATA_PATH = os.path.join(os.getcwd(), 'datasets')

EVALUATION_DICT = {"Accuracy": accuracy_score, "Precision": precision_score, "Recall": recall_score,
                   "F1-score": f1_score, "ROC_AUC": roc_auc_score}

MODEL_DICT = {"MyID3": MyID3, "DecisionTreeClassifier": DecisionTreeClassifier,
              "MyBaggingID3": MyBaggingID3, "BaggingClassifier": BaggingClassifier}


# 7. Evaluation
def evaluate(X, y, data_name, repetitions=2, n_folds=5, sync=False):
    """
    Compares the performance of MyID3 and MyBaggingID3 models against sklearn DecisionTreeClassifier and
    BaggingClassifier respectively use repeated K-fold cross-validation to evaluate both algorithms,
    Using at least 2 repetitions and at least 5 folds and returns Evaluation metrics

    Args:
        X (ndarray):        Preprocessed binary Data matrix of shape(n_samples, n_features)
        y (array like):     list or ndarray with n_samples containing binary target variable
        data_name (str):    Used to log data in wandb package
        repetitions (int):  number of repetitions >= 2
        n_folds (int):      number of folds >= 5
        sync (bool):        a boolean arg determines if the run will be uploaded to wandb project

    Returns:
        evaluation_dict (dict): Dictionary of evaluation scores - Accuracy, Precision, Recall, F1-score, ROC_AUC score
    """
    assert (len(np.unique(y)) == 2 and len(np.unique(X)) == 2)
    assert (repetitions >= 2 and n_folds >= 5)

    evaluation_dict = dict()
    # Perform the evaluation process for each of the models
    for model_name, model_class in MODEL_DICT.items():
        print(f"Starting evaluation process on {model_name}")
        if sync:
            wandb.init(project='MachineLearning_Assignment1', name=f"{data_name}_{model_name}",
                       config={"Dataset": data_name, "Method": model_name})
        evaluation_dict[model_name] = dict()
        # Cross validation method - some of the datasets have class imbalance so Stratified RepeatedKFold is the most appropriate
        rskf = RepeatedKFold(n_splits=n_folds, n_repeats=repetitions, random_state=42)
        model = model_class()
        # Perform cross validation on the tuned model to get the required metrics for the model
        scoring = cross_validate(model, X, y, cv=rskf, return_estimator=True, return_train_score=False,
                                 scoring={name: make_scorer(metric) for name, metric in EVALUATION_DICT.items()})
        for metric, scores in scoring.items():
            if metric == 'estimator':
                continue
            evaluation_dict[model_name][metric] = np.mean(scores)
            if 'time' in metric:  # Transform fit & train fit_time to units of ms
                evaluation_dict[model_name][metric] *= 1000

            # log relevant metrics using the weights & biases package
            if sync and metric != 'score_time' and ('test_' in metric or 'fit' in metric):
                wandb.summary[metric.split('test_')[-1]] = round(evaluation_dict[model_name][metric], 3)

        # Randomly select a decision tree model to plot from the best estimator of the gridsearch
        plot_model = scoring['estimator'][np.random.choice(len(scoring['estimator']), 1)[0]]
        if sync:
            # plot ROC and PR plots in weights & biases and close th model instance
            wandb.sklearn.plot_roc(y, plot_model.predict_proba(X))
            wandb.sklearn.plot_precision_recall(y, plot_model.predict_proba(X))
            wandb.finish()
    return evaluation_dict


def preprocess(X, cat_cols, cont_cols, n_bins=2):
    """
    Preprocessing helper function
        Args:
        X (ndarray):        Original array with data
        cat_cols (list):    list of strings of categorical column names to be oneHot encoded
        cont_cols (list):   list of strings of numerical (continuous) column names to be binned
        n_bins (int):       number of bins for discretization

    Returns:
        X_preprocessed (ndarray): X transformed by the preprocessor class
    """

    # Define the transformers for the preprocessing pipeline
    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    cont_transformer = KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="uniform")
    preprocessor = ColumnTransformer(transformers=[("cat", cat_transformer, cat_cols),
                                                   ("cont", cont_transformer, cont_cols)])

    # Fit and transform the preprocessing pipeline
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed
