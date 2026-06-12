import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

# CONFIGURATION

DATA_PATH = "../dataset/midas_cleaned.csv"
TOP_K_FEATURES = 300
RANDOM_STATE = 42

os.makedirs("../trained_models", exist_ok=True)

# LOAD DATA

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# REMOVE KNOWN LEAKAGE COLUMNS

LEAKAGE_COLUMNS = [
    "F3912",
    "F3918",
    "F3919",
    "F3920",
    "F3921",
    "F3922",
    "F3923"
]

existing_cols = [
    col for col in LEAKAGE_COLUMNS
    if col in df.columns
]

print("\nRemoving leakage columns:")
print(existing_cols)

df.drop(
    columns=existing_cols,
    inplace=True,
    errors="ignore"
)

# CORRELATION CHECK


print("\nChecking highly correlated features...\n")

target_corr = []

for col in df.columns:
    if col == "F3924":
        continue

    try:
        corr = abs(
            df[col].corr(df["F3924"])
        )

        if corr > 0.50:
            target_corr.append(
                (col, corr)
            )

    except:
        pass

target_corr = sorted(
    target_corr,
    key=lambda x: x[1],
    reverse=True
)

print("Features with correlation > 0.50")

for feature, corr in target_corr:
    print(feature, round(corr, 4))


# FEATURES / TARGET

X = df.drop("F3924", axis=1)
y = df["F3924"]

print("\nDataset Shape:", df.shape)

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)


# VARIANCE FILTER

selector = VarianceThreshold(
    threshold=0.001
)

X_train_var = selector.fit_transform(X_train)
X_test_var = selector.transform(X_test)

print(
    "\nAfter Variance Threshold:",
    X_train_var.shape
)

# ==================================================
# FEATURE SELECTION
# ==================================================

print("\nRunning Feature Selection...")

feature_selector = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="logloss"
)

feature_selector.fit(
    X_train_var,
    y_train
)

importance = feature_selector.feature_importances_

selected_feature_names = X_train.columns[
    selector.get_support()
]

top_indices = np.argsort(
    importance
)[-TOP_K_FEATURES:]

top_feature_names = (
    selected_feature_names[top_indices]
)

feature_df = pd.DataFrame({
    "feature": top_feature_names,
    "importance": importance[top_indices]
})

feature_df = feature_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 20 Features\n")
print(feature_df.head(20))

feature_df.to_csv(
    "../trained_models/top_features.csv",
    index=False
)

# TOP FEATURE DATASET


X_train_selected = X_train_var[:, top_indices]
X_test_selected = X_test_var[:, top_indices]

print(
    "\nSelected Train Shape:",
    X_train_selected.shape
)

print(
    "Selected Test Shape :",
    X_test_selected.shape
)

# ==================================================
# XGBOOST MODEL (NO SMOTE)
# ==================================================

print("\nTraining XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train_selected,
    y_train
)

xgb_pred = xgb_model.predict(
    X_test_selected
)

xgb_prob = xgb_model.predict_proba(
    X_test_selected
)[:, 1]

print("\n===== XGBOOST RESULTS =====\n")

print(
    classification_report(
        y_test,
        xgb_pred
    )
)

print(
    confusion_matrix(
        y_test,
        xgb_pred
    )
)

xgb_pr_auc = average_precision_score(
    y_test,
    xgb_prob
)

print(
    f"\nXGBoost PR-AUC: {xgb_pr_auc:.4f}"
)


# LOGISTIC REGRESSION BASELINE

print("\nTraining Logistic Regression...")

scaler = StandardScaler()

X_train_lr = scaler.fit_transform(
    X_train_selected
)

X_test_lr = scaler.transform(
    X_test_selected
)

lr = LogisticRegression(
    max_iter=10000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

lr.fit(
    X_train_lr,
    y_train
)

lr_pred = lr.predict(
    X_test_lr
)

print("\n===== LOGISTIC REGRESSION RESULTS =====\n")

print(
    classification_report(
        y_test,
        lr_pred
    )
)

# SAVE MODEL

joblib.dump(
    xgb_model,
    "../ML/midas_xgboost.pkl"
)

np.save(
    "../ML/top_feature_indices.npy",
    top_indices
)

print("\nModel Saved Successfully")

