from typing import Dict, List, Tuple
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import os
import froodo as fr

num_workers = 2
data_path = Path("/local/scratch/NCT-CRC-HE/NCT-CRC-HE-100K/")
batch_size = 32
total_samples, num_samples = 0, 0
seed = 42
train_client_indices = [0,1,2,3,4,5,6,7]
ood_client_indices = []
num_clients = 10
num_test_clients = 2


def custom_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')
    
# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx



class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter

    def _artifact_list(self):
        
        darkspots = fr.DarkSpotsAugmentation(sample_intervals=[(3, 5)],scale=2,keep_ignorred=True)
        fatspots  = fr.FatAugmentation(sample_intervals=[(1., 5)],scale=2, keep_ignorred=True)
        squamous  = fr.SquamousAugmentation(sample_intervals=[(2, 3)],scale=2, keep_ignorred=True)
        thread    = fr.ThreadAugmentation(sample_intervals=[(2, 4)],scale=2, keep_ignorred=True)
        blood     = fr.BloodCellAugmentation(sample_intervals=[(1, 25)],scale=3,scale_sample_intervals=[(1.0, 1.02)])
        blood.scale = 0.1
        bubble    = fr.BubbleAugmentation(base_augmentation=transforms.GaussianBlur(kernel_size=(9, 9),sigma=10))
        bubble.overlay_h = 700
        bubble.overlay_w = 700

        artifact_list = [darkspots, fatspots, squamous, thread, blood, bubble]

        return artifact_list



    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        
        self.paths = list(Path(targ_dir).glob("*/*.tif")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms

        self.train_client_indices = train_client_indices
        self.ood_client_indices = ood_client_indices
        self.center = []

        total_samples =len(self.paths)
        num_samples = total_samples
        num_samples_per_client = int(num_samples/num_clients)

        for i in range(num_clients):
            self.center += [i]*num_samples_per_client
        
        self.center = np.random.permutation(self.center)


        # try:
        #     # print("whyg")
        #     self.center = torch.load(Path("../center.pt"))
        # except:
        #     regex = re.compile(r'patch_patient_(\d+)_node_(\d+)_x_(\d+)_y_(\d+).png')
        #     for idx in tqdm(range(len(self.paths))):
        #         image_path = self.paths[idx]
        #         mo=regex.search(str(image_path)[69:])
        #         patient,node,x,y = mo.groups()
        #         patient,node,x,y = int(patient), int(node), int(x), int(y)
        #         self.center.append(int(df[(df["patient"] == patient) & (df["node"] == node) & (df["x_coord"] == x) & (df["y_coord"] == y)]["center"].iloc[0]))

        self.aug_types = self._artifact_list()
        # self.augs = []
        # for i in range(len(self.center)):
        #     if self.center[i] in self.ood_client_indices:
        #         self.augs.append(np.random.choice(self.aug_types))
        #     else:
        #         self.augs.append(None)
        # self.augs = [random.choice(self.aug_types) if self.center[i] in self.ood_client_indices else None for i in range(len(self.center))]
                    # for idx in range(len(self.paths)):
        #     image_path = self.paths[idx]
        #     regex = re.compile(r'patch_patient_(\d+)_node_(\d+)_x_(\d+)_y_(\d+).png')
        #     mo=regex.search(str(image_path)[69:])
        #     patient,node,x,y = mo.groups()
        #     patient,node,x,y = int(patient), int(node), int(x), int(y)
        #     self.center[idx] = int(df[(df["patient"] == patient) & (df["node"] == node) & (df["x_coord"] == x) & (df["y_coord"] == y)]["center"].iloc[0])


        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        img = Image.open(image_path)
        img_tensor = (torch.tensor(np.asarray(img).astype(np.float32)/255)).permute(2,0,1)[:3]
        if self.center[index] in self.ood_client_indices:
            aug = np.random.choice(self.aug_types)
            modified_img_tensor = (aug(fr.Sample(img_tensor))).image
            modified_img_array = (modified_img_tensor.permute(1,2,0) * 255).numpy().astype(np.uint8)
        else:
            modified_img_array = (img_tensor.permute(1,2,0) * 255).numpy().astype(np.uint8)
        return Image.fromarray(modified_img_array)
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        img = img.convert("RGB")
        img_arr = np.asarray(img)
        img = Image.fromarray(img_arr)
        image_path = self.paths[index]
        # regex = re.compile(r'patch_patient_(\d+)_node_(\d+)_x_(\d+)_y_(\d+).png')
        # mo=regex.search(str(image_path)[69:])
        # # print(mo.groups())
        # patient,node,x,y = mo.groups()
        # patient,node,x,y = int(patient), int(node), int(x), int(y)

        # has_cancer = int(self.df[(self.df["patient"] == patient) & (self.df["node"] == node) & (self.df["x_coord"] == x) & (self.df["y_coord"] == y)]["tumor"].iloc[0])     
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)


def create_dataloaders(data_transform: transforms.Compose):
    # random.seed(seed)
    np.random.seed(seed)

    data = ImageFolderCustom(
        targ_dir = data_path,
        transform = data_transform
    )
    # print(len(paths))
    # # Setup transforms
    # df = pd.read_csv("/local/scratch/camelyon17/camelyon17_v1.0/metadata.csv")
    # regex = re.compile(r'patch_patient_(\d+)_node_(\d+)_x_(\d+)_y_(\d+).png')
    # for idx in tqdm(range(len(paths))):
    #     image_path = paths[idx]
    #     mo=regex.search(str(image_path)[69:])
    #     patient,node,x,y = mo.groups()
    #     patient,node,x,y = int(patient), int(node), int(x), int(y)
    #     # center.append(int(df[(df["patient"] == patient) & (df["node"] == node) & (df["x_coord"] == x) & (df["y_coord"] == y)]["center"].iloc[0]))

    indices_subsets = {}
    for i in range(num_clients):
        indices_subsets[i] = []

    for i in range(len(data.center)):
        indices_subsets[data.center[i]].append(i)

    for (key, value) in indices_subsets.items():
        indices_subsets[key] = np.random.permutation(indices_subsets[key])


    total_samples = len(data)
    num_samples = total_samples

    train_indices=[]
    validate_indices=[]
    test_indices=[]

    train_client_indices = data.train_client_indices
    # ood_client_indices = [3]

    for i in range(num_clients):
        if(i in train_client_indices):
            train_split = int(0.8*len(indices_subsets[i]))
            train_indices.append(indices_subsets[i][0:train_split])
            validate_indices.append(indices_subsets[i][train_split:])
        else:
            test_indices.append(indices_subsets[i])

    train_datasets = []
    validate_datasets = []
    test_datasets = []

    tr_idx, te_idx = 0,0
    for i in range(num_clients):
        if(i in train_client_indices):
            train_datasets.append(Subset(data, train_indices[tr_idx]))
            validate_datasets.append(Subset(data, validate_indices[tr_idx]))
            tr_idx+=1
        else:
            test_datasets.append(Subset(data, test_indices[te_idx]))
            te_idx+=1
    
    train_dataloaders = []
    validate_dataloaders = []
    test_dataloaders = []

    tr_idx, te_idx = 0,0


    for i in range(num_clients):
        if(i in train_client_indices):
            train_dataloaders.append(
                DataLoader(
                    dataset = train_datasets[tr_idx],
                    batch_size = batch_size,
                    shuffle = True,
                    num_workers = num_workers,
                    pin_memory = False
            ))

            validate_dataloaders.append(
                DataLoader(
                    dataset = validate_datasets[tr_idx],
                    batch_size = batch_size,
                    shuffle = True,
                    num_workers = num_workers,
                    pin_memory = False
                )
            )

            tr_idx += 1

        else:
            test_dataloaders.append(
                DataLoader(
                    dataset = test_datasets[te_idx],
                    batch_size = batch_size,
                    shuffle = True,
                    num_workers = num_workers,
                    pin_memory = False
                )
            )
            te_idx += 1

    return train_dataloaders, validate_dataloaders, test_dataloaders

























# #TODO: fix this file and make it similar to data_setup_ood.py
# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Subset
# from pathlib import Path
# import random
# import numpy as np

# num_workers = 2
# data_path = Path("/local/scratch/NCT-CRC-HE/NCT-CRC-HE-100K/")
# batch_size = 32
# total_samples, num_samples = 0, 0
# seed = 42
# num_clients = 5
# num_test_clients = 1

# def create_dataloaders(data_transform: transforms.Compose):
#     random.seed(seed)

#     data = datasets.ImageFolder(
#         root = data_path,
#         transform = data_transform
#     )

#     total_samples =len(data)
#     num_samples = total_samples

#     indices = torch.tensor(np.random.permutation(np.arange(total_samples)))

#     train_indices=[]
#     validate_indices=[]
#     test_indices=[]

#     num_samples_per_client = int(num_samples/num_clients)
#     train_split, val_split = int(0.8*num_samples_per_client), num_samples_per_client - int(0.8*num_samples_per_client)

#     for i in range(num_clients):
#         if(i < num_clients - num_test_clients):
#             train_indices.append(indices[num_samples_per_client*(i): num_samples_per_client*(i) + train_split])
#             validate_indices.append(indices[num_samples_per_client*(i) + train_split: num_samples_per_client*(i+1)])
#         else:
#             test_indices.append(indices[num_samples_per_client*(i): num_samples_per_client*(i + 1)])

#     train_datasets = []
#     validate_datasets = []
#     test_datasets = []

#     for i in range(num_clients):
#         if(i < num_clients - num_test_clients):
#             train_datasets.append(Subset(data, train_indices[i]))
#             validate_datasets.append(Subset(data, validate_indices[i]))
#         else:
#             test_datasets.append(Subset(data, test_indices[i - (num_clients - num_test_clients)]))
    
#     train_dataloaders = []
#     validate_dataloaders = []
#     test_dataloaders = []

#     for i in range(num_clients):
#         if(i < num_clients - num_test_clients):
#             train_dataloaders.append(
#                 DataLoader(
#                     dataset = train_datasets[i],
#                     batch_size = batch_size,
#                     shuffle = True,
#                     num_workers = num_workers,
#                     pin_memory = True
#             ))

#             validate_dataloaders.append(
#                 DataLoader(
#                     dataset = validate_datasets[i],
#                     batch_size = batch_size,
#                     shuffle = True,
#                     num_workers = num_workers,
#                     pin_memory = True
#                 )
#             )
#         else:
#             test_dataloaders.append(
#                 DataLoader(
#                     dataset = test_datasets[i - (num_clients - num_test_clients)],
#                     batch_size = batch_size,
#                     shuffle = True,
#                     num_workers = num_workers,
#                     pin_memory = True
#                 )
#             )

#     return train_dataloaders, validate_dataloaders, test_dataloaders
    
