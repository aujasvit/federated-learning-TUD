import server_new
from pathlib import Path
import torch
import torchvision
import data_setup_ood_severe_10



device = "cuda:3" if torch.cuda.is_available() else "cpu"

def create_random_model():
    weights = torchvision.models.DenseNet121_Weights.DEFAULT
    model = torchvision.models.densenet121(weights=weights).to(device)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features = 1024, out_features = 2)
    ).to(device)
    return model

weights = torchvision.models.DenseNet121_Weights.DEFAULT
transform = weights.transforms()

new_server = server_new.Server(
    num_train_clients=8,
    num_test_clients=2,
    loss_fn_type = torch.nn.CrossEntropyLoss,
    optimizer_type=torch.optim.Adam,
    model_name="DenseNet-121-untrained",
    experiment_name="new-10-client-1-server-10-ood-trained-Adam",
    device=device,
    lr=0.001,
    create_random_model=create_random_model,
    data_loader_function=data_setup_ood_severe_10.create_dataloaders,
    transform=transform
)


new_server.run(
    server_epochs=10,
    client_epochs=1,
    folder_path=Path("../new-paradigm/new-10-client-1-server-10-ood-trained-Adam")
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


# from pathlib import Path
# import torch 
# import torch.nn as nn
# import os
# import copy
# import numpy as np
# import torchvision

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# weights = torchvision.models.DenseNet121_Weights.DEFAULT
# model = torchvision.models.densenet121(weights=weights)
# auto_transform = weights.transforms()

# model.classifier = torch.nn.Sequential(
#     torch.nn.Linear(in_features = 1024, out_features = 9)
# )

# from tqdm.auto import tqdm

# path = Path("../FedAvg-client-1-server-10-center-ood/models")

# distribution_clients = [0,1]
# test_clients = [2, 3]

# client_paths = [path/f"client-{i}" for i in range(4)]

# fin_mahal = []
# valid_keys = []

# server_path = path/"server"
# _,_,files = next(os.walk(client_paths[0]))
# epochs = len(files)
# for i in client_paths:
#     _,_,files = next(os.walk(i))
#     assert(len(files) == epochs)

# for ep in range(5):
#     client_models = [copy.deepcopy(model) for j in range(len(client_paths))]
#     for i in range(len(client_models)):
#         client_models[i].load_state_dict(torch.load(client_paths[i]/f"epoch-{ep}.pt", map_location = device))
#     server_model = copy.deepcopy(model)
#     server_model.load_state_dict(torch.load(server_path/f"epoch-{ep}.pt", map_location = device))



#     flatten_layer = nn.Flatten(start_dim=0)
#     mahal_dis= {}
#     for i in test_clients:
#         mahal_dis[i] = []

#     # tt = []

#     # pbar = tqdm(total = len(client_models[0].state_dict()), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
#     # d = True
#     # l = 0



#     # l = list(client_models[0].state_dict().items())[173]
#     # print(l[1].shape)

#     for (key, value) in tqdm(client_models[0].state_dict().items(), ncols=0):
#         # print(l + 1, end=" ")
#         # l += 1
#         # if(l <= 172):
#         #     continue
#         # if d:
#         #     print("hi")
#         #     d = False
#         # pbar.update(1)
#         try:
#             torch.cuda.empty_cache()
#             if value.shape == torch.Size([]):
#                 # print("staticmethod")
#                 continue
#             data = np.array([flatten_layer(client_models[i].state_dict()[key]).numpy() for i in distribution_clients])
#             # data = data.astype(np.float32) + 0.00001*np.random.rand(*data.shape)
#             data = torch.tensor(data).to(device)
#             # print(data)
#             # tt = data
#             # break
#             y_mu = torch.mean(data, axis = 0).to(device)
#             cov = torch.cov(data.T).to(device)
#             # print(cov.shape)
#             # break
#             test_data = np.array([flatten_layer(client_models[i].state_dict()[key]).numpy() for i in test_clients])
#             test_data = test_data.astype(np.float32)
#             test_data = torch.tensor(test_data).to(device)
#             test_new = (test_data - y_mu.unsqueeze(0)).to(device)
#             torch.cuda.empty_cache()
#             # left = np.matmul(test_new, inv_cov)
#             # mahal = np.matmul(left, test_new.T)
            
#             mahal = torch.mm(test_new, torch.mm(torch.linalg.pinv(cov), test_new.T))
#             for i in range(len(test_clients)):
#                 mahal_dis[test_clients[i]].append(mahal[i][i].item())

#         except:
#             torch.cuda.empty_cache()
#             continue
    
#     fin_mahal.append(mahal_dis)
    

# torch.save(fin_mahal, Path("./mahal-dis-ood-cmp-five.pt"))
