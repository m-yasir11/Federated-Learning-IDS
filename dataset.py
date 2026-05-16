import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLS = ["proto", "service", "state"]

def load_unsw(path):
    df = pd.read_csv(path)

    # Keep attack category for multi-class
    y_raw = df["attack_cat"]
    y = y_raw.astype("category").cat.codes

    label_mapping = dict(enumerate(y_raw.astype("category").cat.categories))

    # Drop unnecessary columns
    drop_cols = ["id", "label"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ✅ KEEP categorical → encode them
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns], drop_first=True)

    X = df.drop(columns=["attack_cat"], errors="ignore")

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y, label_mapping


def preprocess_unsw(train_path, test_path):
    X_train, y_train, label_map = load_unsw(train_path)
    X_test, y_test, _ = load_unsw(test_path)

    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler, X_train.columns, label_map