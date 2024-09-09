import server_sfa
from pathlib import Path
import torch
import torchvision
import data_setup_center_10

device = "cuda:0" if torch.cuda.is_available() else "cpu"
sfa_clients = [2,6]

def create_random_model():
    model = torchvision.models.densenet121(weights=None).to(device)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features = 1024, out_features = 2)
    ).to(device)

    return model

transform = torchvision.transforms.ToTensor()


new_server = server_sfa.ServerSFA(
    num_train_clients=8,
    num_test_clients=2,
    loss_fn_type = torch.nn.CrossEntropyLoss,
    optimizer_type=torch.optim.Adam,
    model_name="DenseNet-121",
    experiment_name="new-10-client-1-server-10-sfa-untrained-Adam",
    device=device,
    lr=0.001,
    data_loader_function=data_setup_center_10.create_dataloaders,
    create_random_model=create_random_model,
    sfa_clients=sfa_clients,
    transform=transform
)


new_server.run(
    server_epochs=10,
    client_epochs=1,
    folder_path=Path("../new-paradigm/new-10-client-1-server-10-sfa-untrained-Adam")
)




# import server
# from pathlib import Path
# import torch
# import torchvision

# device = "cuda:1" if torch.cuda.is_available() else "cpu"

# weights = torchvision.models.DenseNet121_Weights.DEFAULT
# model = torchvision.models.densenet121(weights=weights).to(device)
# auto_transform = weights.transforms()

# model.classifier = torch.nn.Sequential(
#     torch.nn.Linear(in_features = 1024, out_features = 9)
# )



# new_server = server.Server(
#     num_train_clients=8,
#     num_test_clients=2,
#     loss_fn_type = torch.nn.CrossEntropyLoss,
#     optimizer_type=torch.optim.Adam,
#     model_name="DenseNet-121",
#     experiment_name="FedAvg-10-clients-1-server-10",
#     device=device,
#     lr=0.001,
#     model=model,
#     transform=auto_transform
# )


# new_server.run(
#     server_epochs=5,
#     client_epochs=1,
#     folder_path=Path("../FedAvg-10-clients-1-server-10")
# )