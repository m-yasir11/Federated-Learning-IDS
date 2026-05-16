import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch

def create_clients(X, y, num_clients=4, batch_size=512):
    """
    Split data into non-IID client datasets
    """
    data = np.c_[X, y]
    np.random.shuffle(data)

    splits = np.array_split(data, num_clients)
    clients = []

    for split in splits:
        Xc = torch.tensor(split[:, :-1], dtype=torch.float32)
        yc = torch.tensor(split[:, -1], dtype=torch.long)

        dataset = TensorDataset(Xc, yc)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        clients.append(loader)

    return clients
