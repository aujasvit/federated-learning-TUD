import torch
import torchvision
import os
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


dirs = [
        # "../new-paradigm/new-10-client-1-server-10-ana-75-untrained-Adam/models",
        "../new-paradigm/new-10-client-1-server-10-ood-2-dark-untrained-Adam/models",
        "../new-paradigm/new-10-client-1-server-10-ood-2-severe-untrained-Adam/models",
        "../new-paradigm/new-10-client-1-server-10-ood-2-untrained-Adam/models",
        "../new-paradigm/new-10-client-1-server-10-untrained-Adam/models",
        "../new-paradigm/new-10-client-1-server-10-random-untrained-Adam/models"
        ]


class MahalanobisDistance:
    def __init__(self, device1, device2):
        self.device1 = device1
        self.device2 = device2

    def _mahalanobis(self, x_test, x_mean, cov, device):
        # print(cov.shape)
        cov2 = cov.to(device)
        L = torch.linalg.cholesky(cov2)
        test_data = ((x_test - x_mean.unsqueeze(0))).to(device)
        mdis = torch.linalg.solve(L, test_data.T)
        return torch.linalg.norm(mdis, dim=0)
    
    def _mahalanobis_approx(self, x_test, x_mean, cov, device):
        cov2 = cov.to(device)
        test_data = ((x_test - x_mean.unsqueeze(0))).to(device)
        return torch.mm(test_data, torch.mm(torch.linalg.pinv(cov2), test_data.T))

    def mahalanobis(self, train_data, test_data):
        train_data = train_data.to(self.device1)
        test_data = test_data.to(self.device1)
        train_mean = torch.mean(train_data, dim=0)
        
        train_cov = torch.cov(train_data.T)
        if(train_cov.shape == torch.Size([])):
            train_cov = train_cov.reshape(1,1)
        
        return self._mahalanobis(test_data, train_mean, train_cov, self.device2)
    
    def mahalanobis_approx(self, train_data, test_data):
        train_data = train_data.to(self.device1)
        test_data = test_data.to(self.device1)
        train_mean = torch.mean(train_data, dim=0)
        train_cov = torch.cov(train_data.T)
        if(train_cov.shape == torch.Size([])):
            train_cov = train_cov.reshape(1,1)
        return self._mahalanobis_approx(test_data, train_mean, train_cov, self.device2)
    
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



for root_dir in dirs:
    path = Path(root_dir)
    # num_clients = len(next(os.walk(Path(root_dir)/'models'))[1]) - 1
    # print(num_clients)
    client_paths = [path/f"client-{i}" for i in range(8)]
    server_path = path/"server"
    distribution_clients = [0,1,2,3,4,5,6,7]
    test_clients = [0,1,2,3,4,5,6,7]
    server_path = path/"server"
    _,_,files = next(os.walk(server_path))
    num_epochs = len(files)
    device1 = "cuda:0" if torch.cuda.is_available() else 'cpu'
    device2 = "cuda:0" if torch.cuda.is_available() else 'cpu'

    fin_mahal = {}
    for epoch in range(1,num_epochs):
        client_models = [torch.load(client_paths[i]/f"epoch-{epoch}.pt", map_location='cpu') for i in range(len(client_paths))]            
        prev_server_model = torch.load(server_path/f"epoch-{epoch-1}.pt", map_location='cpu')
        keys = list(prev_server_model.keys())
        mahal_dis = {}
        for key in keys:

            mahal_dis[key] = {}

            for i in range(len(client_models)):
                client_models[i][key] = client_models[i][key] - prev_server_model[key]

            temp_m = {}
            used_channels = 0
            try:
                train_data, test_data = [], []

                num_samples_per_client = torch.numel(client_models[0][key])

                for i in distribution_clients:
                    train_data.extend(client_models[i][key].numpy())

                for i in test_clients:
                    test_data.extend(client_models[i][key].numpy())
                    
                        
                train_data = np.array(train_data).astype(np.float64)
                test_data = np.array(test_data).astype(np.float64)
                
                train_data, test_data = torch.tensor(train_data), torch.tensor(test_data)

                
                train_data = train_data.reshape(-1, 1)
                test_data = test_data.reshape(-1, 1)
                # print(train_data.shape, end = " ")
                m = MahalanobisDistance(device1, device2)

                # train_data, test_data, pca = m.pca_transform(train_data=train_data, test_data=test_data, percent_variation=pca_percent_variation)
                # print(train_data.shape, test_data.shape)
                # print(key)
                # print(train_data.shape, test_data.shape)
                # print(pca.n_components_)
                # print(train_data.shape, test_data.shape)
                mahal = m.mahalanobis(train_data, test_data)

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
                        temp_m[client] += torch.sum(mahal[i*num_samples_per_client: (i+1)*num_samples_per_client]).item()/num_samples_per_client
                    else:
                        temp_m[client] = torch.sum(mahal[i*num_samples_per_client: (i+1)*num_samples_per_client]).item()/num_samples_per_client
                
                used_channels += 1
                # print(key)
            except:
                # break
                continue
                # break

            # print(used_channels, client_models[0][key].shape[1])

            for (i, j) in temp_m.items():
                mahal_dis[key][i] = j/used_channels

        fin_mahal[epoch] = mahal_dis
        # break


    itr = 0
    while itr < 20:
        itr += 1
        try:
            torch.save(fin_mahal, path/"../mahal_dis_all_flatten.pt")
        except:
            continue
        else:
            break





# import torch
# from pathlib import Path
# import torchvision
# import data_setup_ood_dark_10
# import client
# import os

# root = Path(f"../new-paradigm/new-10-client-1-server-10-random-untrained-Adam/models")

# device = "cuda:4" if torch.cuda.is_available() else "cpu"
# model = torchvision.models.densenet121(weights=None).to(device)
# auto_transform = torchvision.transforms.ToTensor()
# model.classifier = torch.nn.Sequential(
#         torch.nn.Linear(in_features = 1024, out_features = 2)
#     )

# model = model.to(device)

# total_client_num = len(next(os.walk(root))[1])

# client_num = 0
# _, _, test_dataloader = data_setup_ood_dark_10.create_dataloaders(auto_transform)
# loss_fn = torch.nn.CrossEntropyLoss()
# print(len(test_dataloader))

# # print(total_client_num)

# num_samples = 0
# for i in test_dataloader:
#     num_samples += len(i)



# for client_num in range(total_client_num-1):
#     client_path = root/f"client-{client_num}/"
#     _,_,files = next(os.walk(client_path))
#     num_epochs = len(files)
#     model_path = client_path/f"epoch-{num_epochs-2}.pt"
#     model.load_state_dict(torch.load(model_path, map_location=device))

#     test_loss, test_acc, = 0.,0.

#     for i in test_dataloader:
#         curr_loss, curr_acc = client.test_step(
#             model = model,
#             loss_fn=loss_fn,
#             test_dataloader=i,
#             device = device
#         )
#         test_loss += curr_loss * len(i)/num_samples
#         test_acc += curr_acc * len(i)/num_samples

    
    
#     print(f"Client-{client_num} | Test Loss: {test_loss:.4f} | Test Accuracy: f{test_acc:.4f} ")
#     print(test_loss, test_acc)



# client_path = root/f"server/"
# _,_,files = next(os.walk(client_path))
# num_epochs = len(files)
# model_path = client_path/f"epoch-{num_epochs-2}.pt"
# model.load_state_dict(torch.load(model_path, map_location=device))

# test_loss, test_acc, = 0.,0.

# for i in test_dataloader:
#     curr_loss, curr_acc = client.test_step(
#         model = model,
#         loss_fn=loss_fn,
#         test_dataloader=i,
#         device = device
#     )
#     test_loss += curr_loss * len(i)/num_samples
#     test_acc += curr_acc * len(i)/num_samples


# print(f"Server | Test Loss: {test_loss:.4f} | Test Accuracy: f{test_acc:.4f} ")
















# # import torch
# # from pathlib import Path
# # import torchvision
# # import data_setup_ood
# # import client
# # import os
# # root = Path(f"../new-paradigm/new-10-client-1-server-10-untrained-Adam/models")

# # device = "cuda:0" if torch.cuda.is_available() else "cpu"
# # weights = torchvision.models.DenseNet121_Weights.DEFAULT
# # model = torchvision.models.densenet121(weights=weights).to(device)
# # auto_transform = weights.transforms()
# # model.classifier = torch.nn.Sequential(
# #         torch.nn.Linear(in_features = 1024, out_features = 2)
# #     )

# # model = model.to(device)

# # total_client_num = len(next(os.walk(root))[1])

# # client_num = 0
# # _, _, test_dataloader = data_setup_ood.create_dataloaders(auto_transform)
# # loss_fn = torch.nn.CrossEntropyLoss()

# # # print(total_client_num)

# # # for client_num in range(total_client_num-1):
# # #     client_path = root/f"client-{client_num}/"
# # #     _,_,files = next(os.walk(client_path))
# # #     num_epochs = len(files)
# # #     model_path = client_path/f"epoch-{num_epochs-1}.pt"
# # #     model.load_state_dict(torch.load(model_path))


# # #     test_loss, test_acc = client.test_step(
# # #         model = model,
# # #         loss_fn=loss_fn,
# # #         test_dataloader=test_dataloader[0],
# # #         device = device
# # #     )
# # #     print(f"Client-{client_num} | Test Loss: {test_loss:.4f} | Test Accuracy: f{test_acc:.4f} ")
# # #     # print(test_loss, test_acc)

# # client_path = root/f"server/"
# # _,_,files = next(os.walk(client_path))
# # num_epochs = len(files)
# # model_path = client_path/f"epoch-{num_epochs-1}.pt"
# # model.load_state_dict(torch.load(model_path))
# # test_loss, test_acc, num_samples = 0.0,0.0,0.0
# # for i in test_dataloader:
# #     num_samples += len(i)

# # for i in test_dataloader:
# #     curr_loss, curr_acc = client.test_step(
# #         model = model,
# #         loss_fn=loss_fn,
# #         test_dataloader=test_dataloader[0],
# #         device = device
# #     )
# # #     print(f"Client-{client_num} | Test Loss: {test_loss:.4f} | Test Accuracy: f{test_acc:.4f} ")

# #     test_loss += curr_loss*len(i)/num_samples
# #     test_acc += curr_acc*len(i)/num_samples
# # print(f"Server | Test Loss: {test_loss:.4f} | Test Accuracy: f{test_acc:.4f} ")



# # from pathlib import Path
# # import torch 
# # import torch.nn as nn
# # import os
# # import copy
# # import numpy as np
# # import torchvision

# # device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
# # weights = torchvision.models.DenseNet121_Weights.DEFAULT
# # model = torchvision.models.densenet121(weights=weights)
# # auto_transform = weights.transforms()

# # model.classifier = torch.nn.Sequential(
# #     torch.nn.Linear(in_features = 1024, out_features = 2)
# # )

# # from tqdm.auto import tqdm

# # path = Path("../FedAvg-client-1-server-10-center-take-2/models")

# # distribution_clients = [0,1]
# # test_clients = [2, 3]

# # client_paths = [path/f"client-{i}" for i in range(4)]

# # fin_mahal = []

# # server_path = path/"server"
# # _,_,files = next(os.walk(client_paths[0]))
# # epochs = len(files)
# # for i in client_paths:
# #     _,_,files = next(os.walk(i))
# #     assert(len(files) == epochs)

# # for ep in range(5):
# #     client_models = [copy.deepcopy(model) for j in range(len(client_paths))]
# #     for i in range(len(client_models)):
# #         client_models[i].load_state_dict(torch.load(client_paths[i]/f"epoch-{ep}.pt", map_location = device))
# #     server_model = copy.deepcopy(model)
# #     server_model.load_state_dict(torch.load(server_path/f"epoch-{ep}.pt", map_location = device))



# #     flatten_layer = nn.Flatten(start_dim=0)
# #     mahal_dis= {}
# #     for i in test_clients:
# #         mahal_dis[i] = []

# #     # tt = []

# #     # pbar = tqdm(total = len(client_models[0].state_dict()), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
# #     # d = True
# #     # l = 0



# #     # l = list(client_models[0].state_dict().items())[173]
# #     # print(l[1].shape)

# #     for (key, value) in tqdm(client_models[0].state_dict().items(), ncols=0):
# #         # print(l + 1, end=" ")
# #         # l += 1
# #         # if(l <= 172):
# #         #     continue
# #         # if d:
# #         #     print("hi")
# #         #     d = False
# #         # pbar.update(1)
# #         try:
# #             torch.cuda.empty_cache()
# #             if value.shape == torch.Size([]):
# #                 # print("staticmethod")
# #                 continue
# #             data = np.array([flatten_layer(client_models[i].state_dict()[key]).numpy() for i in distribution_clients])
# #             # data = data.astype(np.float32) + 0.00001*np.random.rand(*data.shape)
# #             data = torch.tensor(data).to(device)
# #             # print(data)
# #             # tt = data
# #             # break
# #             y_mu = torch.mean(data, axis = 0).to(device)
# #             cov = torch.cov(data.T).to(device)
# #             # print(cov.shape)
# #             # break
# #             test_data = np.array([flatten_layer(client_models[i].state_dict()[key]).numpy() for i in test_clients])
# #             test_data = test_data.astype(np.float32)
# #             test_data = torch.tensor(test_data).to(device)
# #             test_new = (test_data - y_mu.unsqueeze(0)).to(device)
# #             torch.cuda.empty_cache()
# #             # left = np.matmul(test_new, inv_cov)
# #             # mahal = np.matmul(left, test_new.T)
            
# #             mahal = torch.mm(test_new, torch.mm(torch.linalg.pinv(cov), test_new.T))
# #             for i in range(len(test_clients)):
# #                 mahal_dis[test_clients[i]].append(mahal[i][i].item())
# #         except:
# #             torch.cuda.empty_cache()
# #             continue
    
# #     fin_mahal.append(mahal_dis)
    

# # torch.save(fin_mahal, Path("./mahal-dis-center-cmp-five.pt"))
