import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

from dataset import preprocess_unsw
from model import IDSModel

X_train, y_train, X_test, y_test, scaler, feature_cols, label_map = preprocess_unsw(
    "data/UNSW_NB15_training-set.csv",
    "data/UNSW_NB15_testing-set.csv"
)

# Save artifacts
joblib.dump(scaler, "saved_model/scaler.pkl")
joblib.dump(feature_cols, "saved_model/features.pkl")
joblib.dump(label_map, "saved_model/label_map.pkl")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=512,
    shuffle=True
)

model = IDSModel(X_train.shape[1], len(label_map))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "saved_model/global_ids_model.pt")
print("Model + preprocessing saved")