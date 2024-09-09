import torch
import torchvision
import os
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


dirs = [
        "../new-paradigm/new-10-client-1-server-10-ood-2-dark-untrained-Adam/models", 
        "../new-paradigm/new-10-client-1-server-10-ood-2-severe-untrained-Adam/models",
        "../new-paradigm/new-10-client-1-server-10-ood-2-untrained-Adam/models",
        "../new-paradigm/new-10-client-1-server-10-random-untrained-Adam/models",
        "../new-paradigm/new-10-client-1-server-10-untrained-Adam/models",
        # "../new-paradigm/new-10-client-1-server-10-ood-2-severe-single-untrained-Adam/models",        
        "../new-paradigm/new-10-client-1-server-10-ana-75-untrained-Adam/models",
        ]

# pca_percent_variation = 0.95


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
        
        train_cov = torch.diag(torch.var(train_data, dim = 0))
        if(train_cov.shape == torch.Size([])):
            train_cov = train_cov.reshape(1,1)
        
        return self._mahalanobis(test_data, train_mean, train_cov, self.device2)
    
    def mahalanobis_approx(self, train_data, test_data):
        train_data = train_data.to(self.device1)
        test_data = test_data.to(self.device1)
        train_mean = torch.mean(train_data, dim=0)
        train_cov = torch.diag(torch.mean(train_data, dim=0))
        if(train_cov.shape == torch.Size([])):
            train_cov = train_cov.reshape(1,1)
        return self._mahalanobis_approx(test_data, train_mean, train_cov, self.device2)
    
    def pca_transform(self, train_data, test_data):
        if torch.is_tensor(train_data):
            train_data = train_data.to('cpu').numpy()
        if torch.is_tensor(test_data):
            test_data = test_data.to('cpu').numpy()

        scalar = StandardScaler()
        scalar.fit(train_data)
        train_data = scalar.transform(train_data)
        test_data = scalar.transform(test_data)

        pca = PCA()
        pca.fit(train_data)
        train_data = pca.transform(train_data)
        test_data = pca.transform(test_data)

        train_data = torch.tensor(train_data).to(self.device1)
        test_data = torch.tensor(test_data).to(self.device2)

        return train_data, test_data, pca


for root_dir in dirs:
    path = Path(root_dir)
    client_paths = [path/f"client-{i}" for i in range(8)]
    server_path = path/"server"
    distribution_clients = [0,1,2,3,4,5,6,7]
    test_clients = [0,1,2,3,4,5,6,7]
    server_path = path/"server"
    _,_,files = next(os.walk(client_paths[0]))
    num_epochs = len(files)
    device1 = "cuda:2" if torch.cuda.is_available() else 'cpu'
    device2 = "cuda:2" if torch.cuda.is_available() else 'cpu'


    fin_mahal = {}
    for epoch in range(1, num_epochs):
        client_models = [torch.load(client_paths[i]/f"epoch-{epoch}.pt", map_location='cpu') for i in range(len(client_paths))]            
        prev_server_model = torch.load(server_path/f"epoch-{epoch-1}.pt", map_location='cpu')
        keys = list(prev_server_model.keys())
        mahal_dis = {}
        for key in keys:
            for i in range(len(client_models)):
                client_models[i][key] = client_models[i][key] - prev_server_model[key]
            if "conv" in key:
            
                mahal_dis[key] = {}
                


                temp_m = {}
                used_channels = 0
                for in_channel in range(client_models[0][key].shape[1]):
                    try:
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
                        m = MahalanobisDistance(device1, device2)

                        train_data, test_data, pca = m.pca_transform(train_data=train_data, test_data=test_data)
                        # train_data, test_data = torch.tensor(train_data), torch.tensor(test_data)
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
                    except:
                        # print(key, in_channel)
                        continue
                    # break

                # print(used_channels, client_models[0][key].shape[1])

                for (i, j) in temp_m.items():
                    mahal_dis[key][i] = j/used_channels
            
            else:
                mahal_dis[key] = {}

                temp_m = {}
                used_channels = 0
                try:
                    train_data, test_data = [], []

                    num_samples_per_client = 1
                    

                    for i in distribution_clients:
                        train_data.extend(client_models[i][key].reshape(1, -1).numpy())

                    for i in test_clients:
                        test_data.extend(client_models[i][key].reshape(1, -1).numpy())
                        
                            
                    train_data = np.array(train_data).astype(np.float64)
                    test_data = np.array(test_data).astype(np.float64)
                    

                    
                    # print(train_data.shape, end = " ")
                    m = MahalanobisDistance(device1, device2)

                    train_data, test_data, pca = m.pca_transform(train_data=train_data, test_data=test_data)
                    # train_data, test_data = torch.tensor(train_data), torch.tensor(test_data)
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
                    
                    # used_channels += 1
                except:
                    # print(key, in_channel)
                    continue
                    # break

                # print(used_channels, client_models[0][key].shape[1])

                for (i, j) in temp_m.items():
                    mahal_dis[key][i] = j


        fin_mahal[epoch] = mahal_dis
        # break

    itr = 0
    while itr < 20:
        itr += 1
        try:
            torch.save(fin_mahal, path/"../mahal_dis_all_pca.pt")
        except:
            continue
        else:
            break

