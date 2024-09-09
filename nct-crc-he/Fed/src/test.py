import torch
from pathlib import Path
import torchvision
import data_setup_ood
import client
import os

root = Path(f"../FedAvg-client-1-server-10-center-ood/models")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
weights = torchvision.models.DenseNet121_Weights.DEFAULT
model = torchvision.models.densenet121(weights=weights).to(device)
auto_transform = weights.transforms()
model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features = 1024, out_features = 9)
    )

model = model.to(device)

total_client_num = len(next(os.walk(root))[1])

client_num = 0
_, test_dataloader, fu = data_setup_ood.create_dataloaders(auto_transform)
loss_fn = torch.nn.CrossEntropyLoss()

# print(total_client_num)

for client_num in range(total_client_num-1):
    client_path = root/f"client-{client_num}/"
    _,_,files = next(os.walk(client_path))
    num_epochs = len(files)
    model_path = client_path/f"epoch-{num_epochs-1}.pt"
    model.load_state_dict(torch.load(model_path))


    test_loss, test_acc = client.validate_step(
        model = model,
        loss_fn=loss_fn,
        validate_dataloader=fu[0],
        device = device
    )
    print(f"Client-{client_num} | Test Loss: {test_loss:.4f} | Test Accuracy: f{test_acc:.4f} ")
    # print(test_loss, test_acc)

    



client_path = root/f"server/"
_,_,files = next(os.walk(client_path))
num_epochs = len(files)
model_path = client_path/f"epoch-{num_epochs-1}.pt"
model.load_state_dict(torch.load(model_path))


test_loss, test_acc = client.test_step(
    model = model,
    loss_fn=loss_fn,
    test_dataloader=fu[0],
    device = device
)
print(f"Server | Test Loss: {test_loss:.4f} | Test Accuracy: f{test_acc:.4f} ")
