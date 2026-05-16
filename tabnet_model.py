import torch
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier


def create_tabnet(input_dim):
    return TabNetClassifier(
        input_dim=input_dim,
        output_dim=2,
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-3),
        mask_type='sparsemax'
    )


# ✅ FIXED
def get_tabnet_parameters(model):
    return [
        param.detach().cpu().numpy()
        for param in model.network.parameters()
    ]


# ✅ FIXED
def set_tabnet_parameters(model, parameters):
    for param, new_param in zip(model.network.parameters(), parameters):
        param.data = torch.tensor(new_param).to(param.device)