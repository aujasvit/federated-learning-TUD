import copy
import torch
import torch.nn as nn
from typing import List

def fed_avg(
        client_models: List[nn.Module],
        train_dataloaders: List[torch.utils.data.DataLoader]):
    
    total_samples = 0
    for i in train_dataloaders:
        total_samples += len(i)
    
    fin_model = copy.deepcopy(client_models[0])
    for (key, value) in fin_model.state_dict().items():
        fin_model.state_dict()[key].copy_(value* len(train_dataloaders[0])/total_samples)
        
    for i in range(1, len(client_models)):
        for(key, value) in fin_model.state_dict().items():
            fin_model.state_dict()[key].copy_((client_models[i].state_dict()[key])*(len(train_dataloaders[i])/total_samples) + value)
    
    return fin_model