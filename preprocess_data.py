import os
from PIL import Image
import numpy as np
import pandas as pd
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



class CustomDataset_t(Dataset):
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
        logit = self.dataset[idx][3]
        att = self.dataset[idx][4]
        return id, image, label, logit, att
    
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
        self.logit_path = args.logit_path
        self.att_path = args.att_path
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

        # the path of image and labels file
        self.train_data = self.load_data(flag_train=True)
        self.data = self.load_data()

    def load_data(self, flag_train=False):
        results = []
        id = 0
         
        df_logits = pd.read_csv(self.logit_path, sep='\t')
        df_sorted = df_logits.sort_values(by=df_logits.columns[0], ascending=True)
        logits = df_sorted.iloc[:, 2].apply(lambda x: np.fromstring(x[1:-1], sep=','))
        
        
        df_ig_file = open(self.att_path, "r") 
        df_ig_lines = df_ig_file.readlines()
        ig_features = []
        for line in df_ig_lines:
            _, array_str = line.split(maxsplit=1)
            array_str = array_str[1:-2]
            array = [float(num) for num in array_str.split(',') if num.strip()]
            ig_features.append(array)
        
        for folder_name in os.listdir(self.data_root):
            folder_path = os.path.join(self.data_root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            label = self.mapping.get(folder_name)
            if label is not None:
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    if flag_train:
                        results.append([id, image_path, label, logits.iloc[id-1], ig_features[id]])
                    else:
                        results.append([id, image_path, label])
                id += 1
                    
        return results
    

    def collate_fn_train(self, examples):
        X, Y = [], []
        ID = []

        X_logits = []
        X_att = []
    
        for id, x, y, logit, att in examples:
            ID.append(id)
            X.append(x)
            Y.append(y)
            X_logits.append(logit)
            X_att.append(att)
        

        Y = torch.tensor(Y)
        X_logits = torch.from_numpy(np.array(X_logits))

        return ID, X, Y, X_logits, X_att
    
    def collate_fn(self, examples):
        X, Y = [], []
        ID = []
        for id, x, y in examples:
            ID.append(id)
            X.append(x)
            Y.append(int(y))
            
        Y = torch.tensor(Y)
    
        return ID, X,  Y


    # data_loader
    def get_data_loader(self, data, batch_size, transform=None, shuffle=False, num_workers=4, flag_train=False):
        if flag_train:
            dataset = CustomDataset_t(data, transform)
            data_loader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle, 
                                    num_workers=num_workers,
                                    collate_fn=self.collate_fn_train)
        else:
            dataset = CustomDataset(data, transform)
            data_loader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle, 
                                    num_workers=num_workers,
                                    collate_fn=self.collate_fn)
        return data_loader




