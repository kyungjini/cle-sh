# %%
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBClassifier
import shap


def XGBclf(X_train, X_test, y_train):
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


def make_result(df_X, df_y, str_save_directory):
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

        y_pred_fold, y_prob_fold, shap_fold = XGBclf(X_train, X_test, y_train)
        for _ in shap_fold:
            shap_model.append(_)

        y_pred += y_pred_fold.tolist()
        y_prob += y_prob_fold.tolist()
        y_test += y_test_fold.tolist()

    new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
    df_X_reindex = df_X.reindex(new_index)
    df_X_reindex.to_csv(f"{str_save_directory}/features.csv")

    y_prob = np.array(y_prob)
    fpr, tpr, _ = roc_curve(np.array(y_test), y_prob[:, 1])

    np.save(f"{str_save_directory}/pred", y_pred)
    np.save(f"{str_save_directory}/pred_prob", y_prob)
    np.save(f"{str_save_directory}/shap", np.array(shap_model))

    performance = open(f"{str_save_directory}/performance.txt", "w")
    performance.write(f"AUC: {auc(fpr, tpr)}")
    performance.close()


# %%
####### K-fold ####
kf = KFold(n_splits=5, shuffle=True, random_state=1)

########### IBD, DR, BC #############
datasets = ["IBD", "DR", "BC"]

for dataset in datasets:
    path_save = f"./samples_for_demo/{dataset}/data"  # save directory: 'dataset'/data

    features = np.load(f"{path_save}/data_origin/features.npy")
    label = np.load(f"{path_save}/data_origin/label.npy")

    df_features = pd.DataFrame(
        features,
        columns=[f"feature_{i}" for i in np.arange(1, np.shape(features)[1] + 1)],
    )
    df_label = pd.DataFrame(label, columns=["label"])["label"]

    make_result(df_features, df_label, path_save)

########### HF #############
dataset = "HF"
path_save = f"./samples_for_demo/{dataset}/data"

df = pd.read_csv(f"{path_save}/data_origin/heart_failure.csv")
df["age"] = df["age"] // 10

df_label = df["DEATH_EVENT"]
df_features = df.drop(columns=["DEATH_EVENT"])

make_result(df_features, df_label, path_save)

########### MS #############
dataset = "MS"
path_save = f"./samples_for_demo/{dataset}/data"

df = pd.read_csv(f"{path_save}/data_origin/metabolic_syndrome.csv")
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

df["Sex"] = np.where(df["Sex"] == "Female", 1, 0)
df["Marital"] = np.where(df["Marital"] == "Married", 1, 0)
races = df["Race"].unique()
for race in races[:-1]:
    df[race] = np.where(df["Race"] == race, 1, 0)

df_label = df["MetabolicSyndrome"]
df_features = df.drop(columns=["seqn", "Race", "MetabolicSyndrome"])

save_directory = f"./samples_for_demo/data/{dataset}"
make_result(df_features, df_label, path_save)
