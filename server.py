import flwr as fl
import torch
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import NUM_CLIENTS, DEVICE
from model import IDSModel
from dataset import preprocess_unsw

print("SERVER STARTING...", flush=True)


# ===============================
# Evaluation Function
# ===============================
def get_eval_fn():

    # Load test data once
    X_train, y_train, X_test, y_test, _, _ = preprocess_unsw(
        "data/UNSW_NB15_training-set.csv",
        "data/UNSW_NB15_testing-set.csv"
    )

    def evaluate(server_round, parameters, config):

        # Load model
        model = IDSModel(X_test.shape[1]).to(DEVICE)

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in params_dict}
        model.load_state_dict(state_dict)

        model.eval()

        # Convert to tensors
        X = torch.tensor(X_test).float().to(DEVICE)
        y = torch.tensor(y_test).long().to(DEVICE)

        # Prediction
        with torch.no_grad():
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)

        y_true = y.cpu().numpy()
        y_pred = preds.cpu().numpy()

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(
            f"[Round {server_round}] "
            f"Acc={acc:.3f} "
            f"Recall={rec:.3f} "
            f"Precision={prec:.3f} "
            f"F1={f1:.3f}",
            flush=True
        )

        return 0.0, {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

    return evaluate


# ===============================
# Strategy
# ===============================
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=get_eval_fn(),
)


# ===============================
# Start Server
# ===============================
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )