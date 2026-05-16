import torch
import pandas as pd
import numpy as np
import joblib
from model import IDSModel

MODEL_PATH = "saved_model/global_ids_model.pt"

def main():
    print("Loading trained IDS model...", flush=True)

    scaler = joblib.load("saved_model/scaler.pkl")
    feature_cols = joblib.load("saved_model/features.pkl")
    label_map = joblib.load("saved_model/label_map.pkl")

    model = IDSModel(len(feature_cols), len(label_map))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    print("Model loaded successfully", flush=True)

    # Load test dataset
    df = pd.read_csv("data/UNSW_NB15_testing-set.csv")

    # -----------------------------
    # SAME PREPROCESSING AS TRAINING
    # -----------------------------

    # Encode categorical
    df = pd.get_dummies(df, columns=["proto", "service", "state"], drop_first=True)

    # Drop unused
    df = df.drop(columns=["id", "label", "attack_cat"], errors="ignore")

    # Align features
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    print("\nDEBUG: Processed test data")
    print(df.head())

    # Scale
    X = scaler.transform(df)
    X = np.nan_to_num(X)

    X = torch.tensor(X, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        outputs = model(X)
        outputs = torch.nan_to_num(outputs)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

    print(f"\nPrediction completed on {len(preds)} samples")

    print("\nClass Distribution:")
    unique, counts = np.unique(preds.numpy(), return_counts=True)

    for u, c in zip(unique, counts):
        print(f"{label_map[u]} : {c}")

if __name__ == "__main__":
    main()