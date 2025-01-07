from torch.utils.tensorboard import SummaryWriter
import torch
import os

num_clients=5


def get_summary_writer(experiment_name: str, 
                  model_name: str)-> torch.utils.tensorboard.writer.SummaryWriter:
    layout = {
        "Client": {
            "train_loss": ["Multiline", [f"train_loss/client-{i}" for i in range(1, num_clients+1)]],
            "validate_loss": ["Multiline", [f"validate_loss/client-{i}" for i in range(1, num_clients+1)] + ["validate_loss/server"]],
            "train_acc": ["Multiline", [f"train_acc/client-{i}" for i in range(1, num_clients+1)]],
            "validate_acc": ["Multiline", [f"validate_acc/client-{i}" for i in range(1, num_clients+1)] + ["validate_acc/server"]]
        },
        "Server": {
            "final_validate_loss": ["Multiline", ["final_validate_loss/client", "final_validate_loss/server"]],
            "final_validate_acc": ["Multiline", ["final_validate_acc/client", "final_validate_acc/server"]]
        }
    }
    log_dir = os.path.join("runs", experiment_name, model_name)
    
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_custom_scalars(layout)

    return writer
