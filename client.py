import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE, EPOCHS, LEARNING_RATE, NUM_CLIENTS
from model import IDSModel
from dataset import preprocess_unsw
from federated_data import create_clients

X_train, y_train, _, _, _, _, label_map = preprocess_unsw(
    "data/UNSW_NB15_training-set.csv",
    "data/UNSW_NB15_testing-set.csv"
)

clients = create_clients(X_train, y_train, NUM_CLIENTS)
num_classes = len(label_map)


class IDSClient(fl.client.NumPyClient):
    def __init__(self, train_loader, input_dim):
        self.model = IDSModel(input_dim, num_classes).to(DEVICE)
        self.train_loader = train_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict(
            {k: torch.tensor(v).to(DEVICE) for k, v in state_dict.items()}
        )

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for _ in range(EPOCHS):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.train_loader.dataset), {}


def start_client(cid):
    client = IDSClient(clients[cid], X_train.shape[1])
    fl.client.start_numpy_client("127.0.0.1:8080", client)