import server_new
from pathlib import Path
import torch
import torchvision
import data_setup

device = "cuda:3" if torch.cuda.is_available() else "cpu"
ana_clients = [6,7]

def create_random_model():
    model = torchvision.models.densenet121(weights=None).to(device)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features = 1024, out_features = 9)
    ).to(device)

    return model

transform = torchvision.transforms.ToTensor()


new_server = server_new.Server(
    num_train_clients=8,
    num_test_clients=2,
    loss_fn_type = torch.nn.CrossEntropyLoss,
    optimizer_type=torch.optim.Adam,
    model_name="DenseNet-121",
    experiment_name="new-10-client-1-server-10-untrained-Adam",
    device=device,
    lr=0.001,
    data_loader_function=data_setup.create_dataloaders,
    create_random_model=create_random_model,
    # ana_clients=ana_clients,
    # ana_scale = 50,
    transform=transform
)


new_server.run(
    server_epochs=10,
    client_epochs=1,
    folder_path=Path("../new-paradigm/new-10-client-1-server-10-untrained-Adam")
)

