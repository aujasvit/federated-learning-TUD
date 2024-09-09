import torch
from pathlib import Path
import numpy as np

device = 'cpu'

dirs = [
    # "../new-paradigm/new-10-client-1-server-10-ood-2-severe-untrained-Adam/models",
    # "../new-paradigm/new-10-client-1-server-10-ana-10-untrained-Adam/models",
    "../new-paradigm/new-10-client-1-server-10-ana-25-untrained-Adam/models",
    # "../new-paradigm/new-10-client-1-server-10-ana-50-untrained-Adam/models",
    # "../new-paradigm/new-10-client-1-server-10-ana-75-untrained-Adam/models",
    # "../new-paradigm/new-10-client-1-server-10-sfa-untrained-Adam/models"
]

mc = "mahal/cosine"
ec = "mahal/euclid"
for root_dir in dirs:
    # try:
        r = Path(root_dir)/".."
        cos = torch.load(r/"cosine_dis_fin.pt", map_location = device)
        euclid = torch.load(r/"euclidean_dis_fin.pt", map_location = device)    
        mahal = torch.load(r/"mahal_dis_all_pca.pt", map_location = device)

        cmp = {}

        for (epoch , val) in mahal.items():
            cmp[epoch] = {mc: {}, ec: {}}
            for (key, value) in val.items():
                cmp[epoch][mc][key] = {}
                cmp[epoch][ec][key] = {}
                
                for (client, num) in value.items():
                    cmp[epoch][mc][key][client] = mahal[epoch][key][client]/cos[epoch][key][client] if cos[epoch][key][client] else None
                    cmp[epoch][ec][key][client] = mahal[epoch][key][client]/euclid[epoch][key][client] if euclid[epoch][key][client] else None

        itr = 0
        while itr < 20:
            itr += 1
            try:
                torch.save(cmp, r/"comparison.pt")
            except:
                # print('why')
                continue
            else:
                break
                
        print(root_dir)

            
    # except:
    #     continue