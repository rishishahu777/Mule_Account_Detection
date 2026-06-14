import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
# CONFIG

DATA_PATH = "../dataset/midas_cleaned.csv"
RANDOM_STATE = 42

# LOAD DATA

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# REMOVE LEAKAGE

drop_cols = [
    "F3912",
    "F3918",
    "F3919",
    "F3920",
    "F3921",
    "F3922",
    "F3923"
]

existing = [
    c for c in drop_cols
    if c in df.columns
]

df.drop(
    columns=existing,
    inplace=True,
    errors="ignore"
)

# SPLIT

X = df.drop("F3924", axis=1)
y = df["F3924"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# VARIANCE THRESHOLD

selector = VarianceThreshold(
    threshold=0.001
)

X_train_var = selector.fit_transform(X_train)
X_test_var = selector.transform(X_test)

print(
    "After Variance Threshold:",
    X_train_var.shape
)

# TRAIN ONLY ON NORMAL ACCOUNTS

X_train_normal = X_train_var[
    y_train == 0
]

print(
    "Normal Training Samples:",
    X_train_normal.shape
)

# ISOLATION FOREST

iso = IsolationForest(
    n_estimators=300,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

print("Training Isolation Forest...")

iso.fit(
    X_train_normal
)

# PREDICT

pred = iso.predict(
    X_test_var
)

# convert:
# -1 -> anomaly -> mule
#  1 -> normal

pred = np.where(
    pred == -1,
    1,
    0
)

# RESULTS

print("\n===== ISOLATION FOREST =====\n")

print(
    classification_report(
        y_test,
        pred
    )
)

print(
    confusion_matrix(
        y_test,
        pred
    )
)
