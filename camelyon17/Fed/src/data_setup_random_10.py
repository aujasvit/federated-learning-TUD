import re
from typing import Dict, List, Tuple
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import random
import numpy as np
import pandas as pd
import pathlib
from PIL import Image

num_workers = 2
data_path = Path("/local/scratch/camelyon17/camelyon17_v1.0/patches")
batch_size = 32
total_samples, num_samples = 0, 0
seed = 42
num_clients = 10
df = pd.read_csv("/local/scratch/camelyon17/camelyon17_v1.0/metadata.csv")
num_test_clients = 2


def custom_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')


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
    classes = ["Non-cancerous", "Cancerous"]
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset

# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.png")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        df = pd.read_csv("/local/scratch/camelyon17/camelyon17_v1.0/metadata.csv")
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
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
        regex = re.compile(r'patch_patient_(\d+)_node_(\d+)_x_(\d+)_y_(\d+).png')
        mo=regex.search(str(image_path)[69:])
    # print(mo.groups())
        patient,node,x,y = mo.groups()
        patient,node,x,y = int(patient), int(node), int(x), int(y)

        has_cancer = int(df[(df["patient"] == patient) & (df["node"] == node) & (df["x_coord"] == x) & (df["y_coord"] == y)]["tumor"].iloc[0])     
        class_name  = "Cancerous" if has_cancer else "Non-cancerous" # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)


def create_dataloaders(data_transform: transforms.Compose):
    random.seed(seed)

    data = ImageFolderCustom(
        targ_dir = data_path,
        transform = data_transform
    )

    total_samples = len(data)
    num_samples = total_samples

    indices = torch.tensor(np.random.permutation(np.arange(total_samples)))

    train_indices=[]
    validate_indices=[]
    test_indices=[]

    num_samples_per_client = int(num_samples/num_clients)
    train_split, val_split = int(0.8*num_samples_per_client), num_samples_per_client - int(0.8*num_samples_per_client)

    for i in range(num_clients):
        if(i < num_clients - num_test_clients):
            train_indices.append(indices[num_samples_per_client*(i): num_samples_per_client*(i) + train_split])
            validate_indices.append(indices[num_samples_per_client*(i) + train_split: num_samples_per_client*(i+1)])
        else:
            test_indices.append(indices[num_samples_per_client*(i): num_samples_per_client*(i + 1)])

    train_datasets = []
    validate_datasets = []
    test_datasets = []

    for i in range(num_clients):
        if(i < num_clients - num_test_clients):
            train_datasets.append(Subset(data, train_indices[i]))
            validate_datasets.append(Subset(data, validate_indices[i]))
        else:
            test_datasets.append(Subset(data, test_indices[i - (num_clients - num_test_clients)]))
    
    train_dataloaders = []
    validate_dataloaders = []
    test_dataloaders = []

    for i in range(num_clients):
        if(i < num_clients - num_test_clients):
            train_dataloaders.append(
                DataLoader(
                    dataset = train_datasets[i],
                    batch_size = batch_size,
                    shuffle = True,
                    num_workers = num_workers,
                    pin_memory = False
            ))

            validate_dataloaders.append(
                DataLoader(
                    dataset = validate_datasets[i],
                    batch_size = batch_size,
                    shuffle = True,
                    num_workers = num_workers,
                    pin_memory = False
                )
            )
        else:
            test_dataloaders.append(
                DataLoader(
                    dataset = test_datasets[i - (num_clients - num_test_clients)],
                    batch_size = batch_size,
                    shuffle = True,
                    num_workers = num_workers,
                    pin_memory = False
                )
            )

    return train_dataloaders, validate_dataloaders, test_dataloaders
    
