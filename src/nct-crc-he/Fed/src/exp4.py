import torch
import torchvision
import os
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class EuclidianDistance:
    def __init__(self, device1, device2):
        self.device1 = device1
        self.device2 = device2

    def euclidianDistance(self, test_data, train_data):
        train_data = train_data.to(self.device1)
        test_data = test_data.to(self.device1)
        n, m = test_data.shape[0], train_data.shape[0]
        train_data = train_data.unsqueeze(1).expand(-1, m, -1)
        test_data = test_data.unsqueeze(0).expand(n, -1, -1)
        squared_diff = (train_data - test_data) ** 2
        euclidean_distances = torch.sqrt(squared_diff.sum(dim=2))
        return euclidean_distances
    
    
    def pca_transform(self, train_data, test_data, percent_variation):
        if torch.is_tensor(train_data):
            train_data = train_data.to('cpu').numpy()
        if torch.is_tensor(test_data):
            test_data = test_data.to('cpu').numpy()

        scalar = StandardScaler()
        scalar.fit(train_data)
        train_data = scalar.transform(train_data)
        test_data = scalar.transform(test_data)

        pca = PCA(percent_variation)
        pca.fit(train_data)
        train_data = pca.transform(train_data)
        test_data = pca.transform(test_data)

        train_data = torch.tensor(train_data).to(self.device1)
        test_data = torch.tensor(test_data).to(self.device2)

        return train_data, test_data, pca


dirs = ["../new-paradigm/new-10-client-1-server-10-ood-2-dark-untrained-Adam/models", 
        "../new-paradigm/new-10-client-1-server-10-ood-2-severe-untrained-Adam/models",
        "../new-paradigm/new-10-client-1-server-10-ood-2-untrained-Adam/models",
        "../new-paradigm/new-10-client-1-server-10-untrained-Adam/models",
        "../new-paradigm/new-10-client-1-server-10-ana-75-untrained-Adam/models"]
for root_dir in dirs:
    path = Path(root_dir)
    client_paths = [path/f"client-{i}" for i in range(8)]
    server_path = path/"server"
    distribution_clients = [0,1,2,3,4,5,6,7]
    test_clients = [0,1,2,3,4,5,6,7]
    server_path = path/"server"
    _,_,files = next(os.walk(client_paths[0]))
    num_epochs = len(files)
    device1 = "cuda:0" if torch.cuda.is_available() else 'cpu'
    device2 = "cuda:0" if torch.cuda.is_available() else 'cpu'


    fin_euclid = {}
    for epoch in range(1, num_epochs):
        client_models = [torch.load(client_paths[i]/f"epoch-{epoch}.pt", map_location='cpu') for i in range(len(client_paths))]            
        prev_server_model = torch.load(server_path/f"epoch-{epoch-1}.pt", map_location='cpu')
        keys = list(prev_server_model.keys())
        euclid_dis = {}
        for key in keys:
            if "conv" not in key:
                continue
            
            euclid_dis[key] = {}
            

            for i in range(len(client_models)):
                client_models[i][key] = client_models[i][key] - prev_server_model[key]

            temp_m = {}
            used_channels = 0
            for in_channel in range(client_models[0][key].shape[1]):
                    train_data, test_data = [], []

                    num_samples_per_client = client_models[0][key].shape[0]

                    for i in distribution_clients:
                        train_data.extend(client_models[i][key][:,in_channel].unsqueeze(dim=1).numpy())

                    for i in test_clients:
                        test_data.extend(client_models[i][key][:,in_channel].unsqueeze(dim=1).numpy())
                        
                            
                    train_data = np.array(train_data).astype(np.float64)
                    test_data = np.array(test_data).astype(np.float64)
                    

                    
                    train_data = train_data.reshape(train_data.shape[0], -1)
                    test_data = test_data.reshape(test_data.shape[0], -1)
                    # print(train_data.shape, end = " ")
                    m = EuclidianDistance(device1, device2)

                    # train_data, test_data, pca = m.pca_transform(train_data=train_data, test_data=test_data, percent_variation=pca_percent_variation)
                    train_data, test_data = torch.tensor(train_data), torch.tensor(test_data)
                    # print(key)
                    # print(train_data.shape, test_data.shape)
                    # print(pca.n_components_)
                    # print(train_data.shape, test_data.shape)
                    euclidDis = m.euclidianDistance(test_data, train_data)
                    euclidDis = euclidDis.sum(dim = 0)/euclidDis.shape[1]

                    # print(key, mahal)

                    

                    # for (key, value) in mahal:
                    #     if key in temp_m.keys():
                    #         temp_m[key] += value
                    #     else:
                    #         temp_m[key] = value
                    # temp = mahal.shape

                    for i in range(len(test_clients)):
                        client = test_clients[i]
                        if i in temp_m.keys():
                            # print(key)
                            temp_m[client] += torch.sum(euclidDis[i*num_samples_per_client: (i+1)*num_samples_per_client]).item()/num_samples_per_client
                        else:
                            temp_m[client] = torch.sum(euclidDis[i*num_samples_per_client: (i+1)*num_samples_per_client]).item()/num_samples_per_client
                    
                    used_channels += 1

            # print(used_channels, client_models[0][key].shape[1])

            for (i, j) in temp_m.items():
                euclid_dis[key][i] = j/used_channels
            # break

        fin_euclid[epoch] = euclid_dis
        # break


    itr = 0
    while itr < 20:
        itr += 1
        try:
            torch.save(fin_euclid, path/"../euclid_dis_all_distribution_inchannelwise.pt")
        except:
            continue
        else:
            break


# import server
# from pathlib import Path
# import torch
# import torchvision
# import data_setup

# device = "cuda:2" if torch.cuda.is_available() else "cpu"

# weights = torchvision.models.DenseNet121_Weights.DEFAULT
# model = torchvision.models.densenet121(weights=weights).to(device)
# auto_transform = weights.transforms()

# model.classifier = torch.nn.Sequential(
#     torch.nn.Linear(in_features = 1024, out_features = 9)
# )



# new_server = server.Server(
#     num_train_clients=4,
#     num_test_clients=1,
#     loss_fn_type = torch.nn.CrossEntropyLoss,
#     optimizer_type=torch.optim.Adam,
#     model_name="DenseNet-121",
#     experiment_name="FedAvg-client-1-server-10",
#     device=device,
#     lr=0.001,
#     model=model,
#     data_loader_function=data_setup.create_dataloaders,
#     transform=auto_transform
# )


# new_server.run(
#     server_epochs=10,
#     client_epochs=1,
#     folder_path=Path("../FedAvg-client-1-server-10")
# )




# # import server
# # from pathlib import Path
# # import torch
# # import torchvision

# # device = "cuda:1" if torch.cuda.is_available() else "cpu"

# # weights = torchvision.models.DenseNet121_Weights.DEFAULT
# # model = torchvision.models.densenet121(weights=weights).to(device)
# # auto_transform = weights.transforms()

# # model.classifier = torch.nn.Sequential(
# #     torch.nn.Linear(in_features = 1024, out_features = 9)
# # )



# # new_server = server.Server(
# #     num_train_clients=8,
# #     num_test_clients=2,
# #     loss_fn_type = torch.nn.CrossEntropyLoss,
# #     optimizer_type=torch.optim.Adam,
# #     model_name="DenseNet-121",
# #     experiment_name="FedAvg-10-clients-1-server-10",
# #     device=device,
# #     lr=0.001,
# #     model=model,
# #     transform=auto_transform
# # )


# # new_server.run(
# #     server_epochs=5,
# #     client_epochs=1,
# #     folder_path=Path("../FedAvg-10-clients-1-server-10")
# # )

# import server_ana
# from pathlib import Path
# import torch
# import torchvision
# import data_setup

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# ana_clients = [6,7]

# def create_random_model():
#     model = torchvision.models.densenet121(weights=None).to(device)
#     model.classifier = torch.nn.Sequential(
#         torch.nn.Linear(in_features = 1024, out_features = 9)
#     ).to(device)

#     return model

# transform = torchvision.transforms.ToTensor()


# new_server = server_ana.ServerANA(
#     num_train_clients=8,
#     num_test_clients=2,
#     loss_fn_type = torch.nn.CrossEntropyLoss,
#     optimizer_type=torch.optim.Adam,
#     model_name="DenseNet-121",
#     experiment_name="new-10-client-1-server-10-ana-75-untrained-Adam",
#     device=device,
#     lr=0.001,
#     data_loader_function=data_setup.create_dataloaders,
#     create_random_model=create_random_model,
#     ana_clients=ana_clients,
#     ana_scale = 75,
#     transform=transform
# )


# new_server.run(
#     server_epochs=10,
#     client_epochs=1,
#     folder_path=Path("../new-paradigm/new-10-client-1-server-10-ana-75-untrained-Adam")
# )




# # import server
# # from pathlib import Path
# # import torch
# # import torchvision

# # device = "cuda:1" if torch.cuda.is_available() else "cpu"

# # weights = torchvision.models.DenseNet121_Weights.DEFAULT
# # model = torchvision.models.densenet121(weights=weights).to(device)
# # auto_transform = weights.transforms()

# # model.classifier = torch.nn.Sequential(
# #     torch.nn.Linear(in_features = 1024, out_features = 9)
# # )



# # new_server = server.Server(
# #     num_train_clients=8,
# #     num_test_clients=2,
# #     loss_fn_type = torch.nn.CrossEntropyLoss,
# #     optimizer_type=torch.optim.Adam,
# #     model_name="DenseNet-121",
# #     experiment_name="FedAvg-10-clients-1-server-10",
# #     device=device,
# #     lr=0.001,
# #     model=model,
# #     transform=auto_transform
# # )


# # new_server.run(
# #     server_epochs=5,
# #     client_epochs=1,
# #     folder_path=Path("../FedAvg-10-clients-1-server-10")
# # )