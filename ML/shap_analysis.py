import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold

print("Loading model...")

# Load Model


model = joblib.load(
"midas_xgboost.pkl"
)

# Load Dataset

df = pd.read_csv(
"../dataset/midas_cleaned.csv"
)

# Remove Leakage Columns

drop_cols = [
"F3912",
"F3918",
"F3919",
"F3920",
"F3921",
"F3922",
"F3923",
"F3924"
]

existing = [
c for c in drop_cols
if c in df.columns
]

X = df.drop(columns=existing)

# Same Variance Threshold

selector = VarianceThreshold(
threshold=0.001
)

X_var = selector.fit_transform(X)

selected_feature_names = X.columns[
selector.get_support()
]

# Load Top Features

top_indices = np.load(
"top_feature_indices.npy"
)

top_feature_names = (
selected_feature_names[top_indices]
)

X_selected = X_var[:, top_indices]

# Small Sample

sample_size = min(100, len(X_selected))

sample_idx = np.random.choice(
len(X_selected),
sample_size,
replace=False
)

X_sample = pd.DataFrame(
X_selected[sample_idx],
columns=top_feature_names
)

print("Sample Shape:", X_sample.shape)

# SHAP

print("Running SHAP...")

explainer = shap.TreeExplainer(
model
)

shap_values = explainer.shap_values(
X_sample
)

# Summary Plot

plt.figure(figsize=(12, 8))

shap.summary_plot(
shap_values,
X_sample,
show=False
)

plt.savefig(
"../trained_models/shap_summary.png",
bbox_inches="tight"
)

print("SHAP summary saved.")
