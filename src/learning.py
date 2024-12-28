import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBClassifier
import shap


def parse_args():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config",
    )
    return parser.parse_args()


def XGBclf(X_train, X_test, y_train, kf):
    model = XGBClassifier()
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


def main():
    print(">> Start Learning")

    args = parse_args()

    with open(args.config, "r") as f:
        path_config = json.load(f)

    K_FOLDS = path_config["K_FOLDS"]
    PATH = path_config["PATH"]

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=1)

    path_save = os.path.join(PATH, "data")

    df_X = pd.read_csv(os.path.join(path_save, "features.csv"))
    df_y = pd.read_csv(os.path.join(path_save, "label.csv")).iloc[:, 0]

    y_pred = []
    y_prob = []
    y_test = []
    shap_model = []

    ix_train, ix_test = [], []
    for train_index, test_index in kf.split(df_X):
        ix_train.append(train_index), ix_test.append(test_index)

    for train_index, test_index in zip(ix_train, ix_test):
        X_train, X_test = df_X.iloc[train_index], df_X.iloc[test_index]
        y_train, y_test_fold = df_y.iloc[train_index], df_y.iloc[test_index]

        y_pred_fold, y_prob_fold, shap_fold = XGBclf(X_train, X_test, y_train, kf)
        for _ in shap_fold:
            shap_model.append(_)

        y_pred += y_pred_fold.tolist()
        y_prob += y_prob_fold.tolist()
        y_test += y_test_fold.tolist()

    new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
    df_X_reindex = df_X.reindex(new_index)
    df_X_reindex.to_csv(os.path.join(path_save, "features.csv"))

    y_prob = np.array(y_prob)
    fpr, tpr, _ = roc_curve(np.array(y_test), y_prob[:, 1])

    np.save(os.path.join(path_save, "pred"), y_pred)
    np.save(os.path.join(path_save, "pred_prob"), y_prob)
    np.save(os.path.join(path_save, "shap"), np.array(shap_model))

    performance = open(os.path.join(path_save, "performance.txt"), "w")
    performance.write(f"AUC: {auc(fpr, tpr)}")
    performance.close()

    print(">> Learning Complete")


if __name__ == "__main__":
    main()
