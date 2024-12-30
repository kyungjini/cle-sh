import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import shap


# Utility functions
def parse_args():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config",
    )
    return parser.parse_args()


def save_results(path_save, y_pred, y_prob, shap_value, metrics):

    np.save(os.path.join(path_save, "pred"), y_pred)
    np.save(os.path.join(path_save, "pred_prob"), y_prob)
    np.save(os.path.join(path_save, "shap"), np.array(shap_value))

    with open(os.path.join(path_save, "performance.txt"), "w") as performance:
        for key, value in metrics.items():
            performance.write(f"{key}: {value}\n")


def calculate_metrics(y_true, y_pred, y_prob):

    auc_score = roc_auc_score(y_true, y_prob[:, 1])
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "AUC": auc_score,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
    }


# Sample models
def XGBclf(X_train, X_test, y_train, kf):
    model = XGBClassifier(random_state=42)
    params = {
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.09],
        "n_estimators": [200],
        "reg_alpha": [0.5, 0.7, 0.9],
        "reg_lambda": [1.3, 1.5, 1.7],
    }
    grid_cv = GridSearchCV(model, param_grid=params, cv=kf, n_jobs=-1)
    grid_cv.fit(X_train, y_train)
    y_pred = grid_cv.predict(X_test)
    y_prob = grid_cv.predict_proba(X_test)

    model_best = grid_cv.best_estimator_
    explainer = shap.TreeExplainer(model_best)
    shap_fold = explainer.shap_values(X_test)

    return y_pred, y_prob, shap_fold


def MLPclf(X_train, X_test, y_train, kf):
    model = MLPClassifier(max_iter=500, random_state=42)
    params = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
        "learning_rate_init": [0.001, 0.01],
    }
    grid_cv = GridSearchCV(model, param_grid=params, cv=kf, n_jobs=-1)
    grid_cv.fit(X_train, y_train)
    y_pred = grid_cv.predict(X_test)
    y_prob = grid_cv.predict_proba(X_test)

    model_best = grid_cv.best_estimator_
    explainer = shap.KernelExplainer(model_best.predict_proba, X_train)
    shap_fold = explainer.shap_values(X_test)

    return y_pred, y_prob, shap_fold


# Main function
def main():
    args = parse_args()

    with open(args.config, "r") as f:
        path_config = json.load(f)

    K_FOLDS = path_config["K_FOLDS"]
    PATH = path_config["PATH"]
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=1)

    path_data = os.path.join(PATH, "data")

    df_X = pd.read_csv(os.path.join(path_data, "features_learning.csv"))
    df_y = pd.read_csv(os.path.join(path_data, "label_learning.csv")).iloc[:, 0]

    ix_train, ix_test = [], []
    for train_index, test_index in kf.split(df_X):
        ix_train.append(train_index), ix_test.append(test_index)

    new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
    df_X_reindex = df_X.reindex(new_index)
    df_X_reindex.to_csv(os.path.join(path_data, "features.csv"))

    models = {
        "XGBClassifier": (XGBclf, os.path.join(PATH, "XGB")),
        "MLPClassifier": (MLPclf, os.path.join(PATH, "MLP")),
    }

    for model_name, (model_func, path_save) in models.items():
        print(f">> {model_name} Learning Start")
        y_pred, y_prob, y_test, shap_value = [], [], [], []

        for train_index, test_index in zip(ix_train, ix_test):
            X_train, X_test = df_X.iloc[train_index], df_X.iloc[test_index]
            y_train, y_test_fold = df_y.iloc[train_index], df_y.iloc[test_index]

            y_pred_fold, y_prob_fold, shap_fold = model_func(
                X_train, X_test, y_train, kf
            )

            shap_value.extend(shap_fold)
            y_pred += y_pred_fold.tolist()
            y_prob += y_prob_fold.tolist()
            y_test += y_test_fold.tolist()

        y_prob = np.array(y_prob)
        y_test = np.array(y_test)

        metrics = calculate_metrics(y_test, np.array(y_pred), y_prob)

        if not os.path.exists(path_save):
            os.makedirs(path_save)
        save_results(path_save, y_pred, y_prob, shap_value, metrics)
        print(f">> {model_name} Learning Complete")
    print(">> Learning Complete")


if __name__ == "__main__":
    main()
