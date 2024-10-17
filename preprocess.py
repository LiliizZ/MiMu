import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import logging

logger = logging.getLogger(__file__)

map_i9 = {
    "00_dog": 0,
    "01_bird": 1, 
    '02_wheeled vehicle': 2,
    "03_reptile": 3, 
    "04_carnivore": 4,
    "05_insect": 5, 
    "06_musical instrument": 6,
    "07_primate": 7,
    "08_fish": 8
}

class CustomDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.data_size = len(dataset)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        id = self.dataset[idx][0]
        image_path = self.dataset[idx][1]
        
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.dataset[idx][2]
        return id, image, label


class ImageNetData(Dataset):
    def __init__(self, args, data_root):
        self.data_root = data_root
        self.batch_size = args.batch_size
        self.mapping = {
                    "00_dog": 0,
                    "01_bird": 1, 
                    '02_wheeled vehicle': 2,
                    "03_reptile": 3, 
                    "04_carnivore": 4,
                    "05_insect": 5, 
                    "06_musical instrument": 6,
                    "07_primate": 7,
                    "08_fish": 8
                }

        # image path and label file
        self.data = self.load_data()

    def load_data(self):
        results = []
        id = 0
        for folder_name in os.listdir(self.data_root):
            folder_path = os.path.join(self.data_root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            label = self.mapping.get(folder_name)
            if label is not None:
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    results.append([id, image_path, label])
                    id += 1
        return results
    

# dataloder
def get_data_loader(data, batch_size, transform=None, shuffle=False, num_workers=4):
    dataset = CustomDataset(data, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader




