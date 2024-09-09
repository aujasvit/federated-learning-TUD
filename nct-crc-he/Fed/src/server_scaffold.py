from typing import Type
import server
import torch
import torch.nn as nn

class ScaffoldServer(server.Server):
    def __init__(self,
        num_train_clients: int, #number of training clients
        num_test_clients: int, #number of test clients out of total clients
        model: Type[nn.Module],
        loss_fn_type: Type[nn.Module],
        optimizer_type: Type[torch.optim.Optimizer],
        model_name,
        experiment_name,
        device,
        transform,
        lr
        ):
        super().__init__(
            num_train_clients=num_test_clients,
            num_test_clients=num_test_clients,
            model=model,
            loss_fn_type=loss_fn_type,
            optimizer_type=optimizer_type,
            model_name=model_name,
            experiment_name=experiment_name,
            device=device,
            lr=lr
        )

        self.
