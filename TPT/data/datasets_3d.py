import math
import os

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import PIL
from PIL import Image
import h5py


class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])
    
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()

datasets_3d = ['ModelNet40', 'ScanObjectNN']

path_dict = {
    # dataset_name: ["image_dir", "json_split_file"]
    "modelnet40": ["jpg", "data/data_splits/split_zhou_OxfordFlowers.json"],
    "ScanObjectNN": ["images", "data/data_splits/split_zhou_Food101.json"],
    "dtd": ["images", "data/data_splits/split_zhou_DescribableTextures.json"],
    "pets": ["", "data/data_splits/split_zhou_OxfordPets.json"],
    "sun397": ["", "data/data_splits/split_zhou_SUN397.json"],
    "caltech101": ["", "data/data_splits/split_zhou_Caltech101.json"],
    "ucf101": ["", "data/data_splits/split_zhou_UCF101.json"],
    "cars": ["", "data/data_splits/split_zhou_StanfordCars.json"],
    "eurosat": ["", "data/data_splits/split_zhou_EuroSAT.json"]
}

def build_3d_fewshot_dataset(set_id, root, transform, mode='train', n_shot=None):
    path_suffix, json_path = path_dict[set_id.lower()]
    image_path = os.path.join(root, path_suffix)
    return BaseJsonDataset(image_path, json_path, mode, n_shot, transform)


class ModelNet40(Dataset):
    """ ModelNet40 dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.dataset_dir = root
        
        self.transform = transform
        
        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        classnames = self.read_classnames(text_file)
        
        if mode == 'train':
            data, label = self.load_data(os.path.join(self.dataset_dir, 'train_files.txt'))
            self.data = self.read_data(classnames, data, label)
        elif mode == 'test':
            data, label = self.load_data(os.path.join(self.dataset_dir, 'test_files.txt'))
            self.data = self.read_data(classnames, data, label)
        
    def read_data(self, classnames, datas, labels):
        items = []
        
        for i, data in enumerate(datas):
            label = int(labels[i])
            classname = classnames[label]
            items.append([data, label])
            
        return items
    
    def load_data(self, data_path):
        all_data = []
        all_label = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(os.path.join(self.dataset_dir, h5_name.strip().split('/')[-1]), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                all_data.append(data)
                all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        
        return all_data, all_label
    
    def read_classnames(self, text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                classname = line.strip()
                classnames[i] = classname
        return classnames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]