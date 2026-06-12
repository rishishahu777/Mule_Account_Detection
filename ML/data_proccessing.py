import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


df = pd.read_csv("dataset/DataSet.csv")

# Remove index column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Remove Empty Columns

df.dropna(axis=1, how='all', inplace=True)

# Remove columns with >95% missing values
missing_percent = df.isnull().mean() * 100

high_missing_cols = missing_percent[
    missing_percent > 95
].index

df.drop(columns=high_missing_cols, inplace=True)

# Handle Date Column

df['F3888'] = pd.to_datetime(
    df['F3888'],
    errors='coerce'
)

df['F3888_year'] = df['F3888'].dt.year
df['F3888_month'] = df['F3888'].dt.month
df['F3888_day'] = df['F3888'].dt.day

df.drop(columns=['F3888'], inplace=True)

# Encode Categorical Columns

cat_cols = [
    'F2230',
    'F3886',
    'F3889',
    'F3890',
    'F3891',
    'F3892',
    'F3893'
]

for col in cat_cols:
    df[col] = df[col].astype(str)

    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Missing Value Imputation

imputer = SimpleImputer(
    strategy='median'
)

df = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# Remove Constant Columns

constant_cols = [
    col
    for col in df.columns
    if df[col].nunique() <= 1
]

df.drop(columns=constant_cols,
        inplace=True)

# Save Clean Dataset


df.to_csv(
    "dataset/midas_cleaned.csv",
    index=False
)

print("Cleaned Dataset Shape:", df.shape)


# Feature / Target Split


X = df.drop('F3924', axis=1)
y = df['F3924']

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)


# Variance Threshold

selector = VarianceThreshold(
    threshold=0.001
)

X_train_var = selector.fit_transform(X_train)
X_test_var = selector.transform(X_test)

print(
    "After Variance Threshold:",
    X_train_var.shape
)
