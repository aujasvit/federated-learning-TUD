import torch
import torchvision
import numpy as np
from pathlib import Path
import data_setup_center_10
import client
import os

dirs = [
    "../new-paradigm/new-10-client-1-server-10-ood-2-severe-untrained-Adam/models",
    "../new-paradigm/new-10-client-1-server-10-ana-10-untrained-Adam/models",
    "../new-paradigm/new-10-client-1-server-10-ana-25-untrained-Adam/models",
    "../new-paradigm/new-10-client-1-server-10-ana-50-untrained-Adam/models",
    "../new-paradigm/new-10-client-1-server-10-ana-75-untrained-Adam/models",
    "../new-paradigm/new-10-client-1-server-10-sfa-untrained-Adam/models",
    # "../new-paradigm/new-10-client-1-server-10-untrained-Adam/models",
    # "../new-paradigm/new-10-client-1-server-10-random-untrained-Adam/models",
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"




for root_dir in dirs:
    print(root_dir)
    root = Path(root_dir)

    cmp = torch.load(root/"../comparison.pt", map_location=device)


    tot, num_tot = 0,0
    conv, num_conv = 0,0
    var, num_var = 0,0
    greater = 0
    mean, num_mean = 0,0
    num_batches, num_num_batches = 0,0
    others, num_others = 0,0

    for(key, value) in cmp.items(): #epochs
        d = value["mahal/cosine"]
        
        for(key, value) in d.items():
            if not value:
                continue
            tmp = 0
            for cl in [0,1,3,4,5,7]:
                tmp += 0 if value[cl] == None else value[cl]
            tmp /= 6
            tmp2 = 0
            for cl in [2,6]:
                tmp2 += 0 if value[cl] == None else value[cl]
            tmp2 /= 2

            if "conv" in key:
                conv += tmp2 if tmp == 0 else tmp2/tmp
                num_conv += 1
            elif "running_var" in key:
                var += tmp2 if tmp == 0 else tmp2/tmp
                num_var += 1
            elif "running_mean" in key:
                mean += tmp2 if tmp == 0 else tmp2/tmp
                num_mean += 1
            elif "num_batches_tracked" in key:
                num_batches += tmp2 if tmp == 0 else tmp2/tmp
                num_batches += 1
            else:
                others += tmp2 if tmp == 0 else tmp2/tmp
                num_others += 1

            tot += tmp2 if tmp == 0 else tmp2/tmp
            num_tot += 1
            if tmp2 >= tmp:
                greater += 1


    print()        
    print('cosine')
    print('greater', greater, '/', num_tot)

    print('conv', 0 if num_conv == 0 else conv/num_conv)
    print('running_var', 0 if num_var == 0 else var/num_var)
    print('running_mean', 0 if num_mean == 0 else mean/num_mean)
    print('num_batches_tracked', 0 if num_num_batches == 0 else num_batches/num_batches)
    print('others', 0 if num_others == 0 else others/num_others)
    print('total', 0 if num_tot == 0 else tot/num_tot)

    tot, num_tot = 0,0
    conv, num_conv = 0,0
    var, num_var = 0,0
    mean, num_mean = 0,0
    greater = 0
    num_batches, num_num_batches = 0,0
    others, num_others = 0,0

    for(key, value) in cmp.items(): #epochs
        d = value["mahal/euclid"]
        
        for(key, value) in d.items():
            if not value:
                continue
            tmp = 0
            for cl in range(6):
                tmp += 0 if value[cl] == None else value[cl]
            tmp /= 6
            tmp2 = 0
            for cl in range(6, 8):
                tmp2 += 0 if value[cl] == None else value[cl]
            tmp2 /= 2

            if "conv" in key:
                conv += tmp2 if tmp == 0 else tmp2/tmp
                num_conv += 1
            elif "running_var" in key:
                var += tmp2 if tmp == 0 else tmp2/tmp
                num_var += 1
            elif "running_mean" in key:
                mean += tmp2 if tmp == 0 else tmp2/tmp
                num_mean += 1
            elif "num_batches_tracked" in key:
                num_batches += tmp2 if tmp == 0 else tmp2/tmp
                num_batches += 1
            else:
                others += tmp2 if tmp == 0 else tmp2/tmp
                num_others += 1

            tot += tmp2 if tmp == 0 else tmp2/tmp
            num_tot += 1
            if tmp2 >= tmp:
                greater += 1


    print()        
    print('euclid')
    print('greater', greater, '/', num_tot)

    print('conv', 0 if num_conv == 0 else conv/num_conv)
    print('running_var', 0 if num_var == 0 else var/num_var)
    print('running_mean', 0 if num_mean == 0 else mean/num_mean)
    print('num_batches_tracked', 0 if num_num_batches == 0 else num_batches/num_batches)
    print('others', 0 if num_others == 0 else others/num_others)
    print('total', 0 if num_tot == 0 else tot/num_tot)

        
    print()

    model = torchvision.models.densenet121(weights=None).to(device)
    auto_transform = torchvision.transforms.ToTensor()
    model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features = 1024, out_features = 2)
        )

    model = model.to(device)
    _, _, test_dataloader = data_setup_center_10.create_dataloaders(auto_transform)
    loss_fn = torch.nn.CrossEntropyLoss()

    # print(total_client_num)

    client_path = root/f"server/"
    _,_,files = next(os.walk(client_path))
    num_epochs = len(files)
    model_path = client_path/f"epoch-{num_epochs-1}.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))


    test_loss, test_acc = client.test_step(
        model = model,
        loss_fn=loss_fn,
        test_dataloader=test_dataloader[0],
        device = device
    )
    print(f"Server | Test Loss: {test_loss:.4f} | Test Accuracy: f{test_acc:.4f} ")

    print()
    print()
    print()