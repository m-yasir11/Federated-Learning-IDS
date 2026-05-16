import torch
import pandas as pd
import numpy as np
import joblib
from model import IDSModel

MODEL_PATH = "saved_model/global_ids_model.pt"

def main():
    print("Loading IDS model...")

    # Load preprocessing artifacts
    scaler = joblib.load("saved_model/scaler.pkl")
    feature_cols = joblib.load("saved_model/features.pkl")
    label_map = joblib.load("saved_model/label_map.pkl")

    # Load model
    model = IDSModel(len(feature_cols), len(label_map))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # =========================
    # LOAD ZEEK FLOWS (FIXED)
    # =========================
    df = pd.read_csv(
        "flows.csv",
        sep="\t",   # 🔥 FIX: Zeek logs are tab-separated
        names=[
            "ts", "proto", "service", "duration",
            "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"
        ]
    )

    print("\nDEBUG: Raw Zeek data")
    print(df.head())

    # =========================
    # CLEAN DATA (VERY IMPORTANT)
    # =========================

    # Replace '-' with 0
    df.replace("-", 0, inplace=True)

    # Convert numeric columns safely
    numeric_cols = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # =========================
    # FEATURE MAPPING
    # =========================
    df.rename(columns={
        "duration": "dur",
        "orig_bytes": "sbytes",
        "resp_bytes": "dbytes",
        "orig_pkts": "spkts",
        "resp_pkts": "dpkts"
    }, inplace=True)

    # =========================
    # ENCODING (MATCH TRAINING)
    # =========================
    df = pd.get_dummies(df, columns=["proto", "service"], drop_first=True)

    # Align with training features
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Final cleaning
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    print("\nDEBUG: Processed input")
    print(df.head())

    # =========================
    # SCALING + SAFETY
    # =========================
    X = scaler.transform(df)

    # 🔥 Fix any NaN / Inf
    X = np.nan_to_num(X)

    X = torch.tensor(X, dtype=torch.float32)

    # =========================
    # PREDICTION
    # =========================
    with torch.no_grad():
        outputs = model(X)

        # 🔥 Prevent NaN propagation
        outputs = torch.nan_to_num(outputs)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

    # =========================
    # OUTPUT
    # =========================
    print("\n=== IDS OUTPUT ===")

    for i in range(len(preds)):
        label_id = preds[i].item()
        label_name = label_map[label_id]
        confidence = probs[i].max().item()

        print(f"Flow {i}: {label_name} (confidence={confidence:.2f})")

        if label_name != "Normal" and confidence > 0.8:
            print(f"⚠️ ALERT: {label_name} attack detected!")

if __name__ == "__main__":
    main()