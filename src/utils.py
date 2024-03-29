import pandas as pd
import torch
from PIL import Image
from datasets import Dataset
from dataclasses import dataclass

@dataclass
class LearningConfig:
    lr: int
    epochs: int
    batch: int
    device: str
    run_name: str
    weight_decay: float
    lr_step: int
    arcface_s: int
    archface_m: float
    scheduler_gamma: float
    backbone: str
    head_name: str
    use_scheduler: str
    

class CustomPeopleDataset(Dataset):
    def __init__(self, data_table_path, processor, base_dir='.'):
        self._data = pd.read_csv(data_table_path, sep=';')

        self.labels_map = {label: i for i, label in enumerate(self._data['label'].unique())}
        self.base_dir = base_dir
        self.processor = processor

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        image_path = f"{self.base_dir}/{self._data['relative_path'][idx]}/{self._data['image_name'][idx]}"
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = torch.tensor(self.processor(image)['pixel_values'][0])
        image.close()
        
        label = self.labels_map[self._data['label'][idx]]

        return image_tensor, label
    
    def __getitems__(self, idxs):
        return [self.__getitem__(idx) for idx in idxs]
        

def custom_collate(data):

    images = torch.cat([torch.unsqueeze(item[0], 0) for item in data], 0)
    labels = torch.tensor([item[1] for item in data])

    return {
        "images": images, 
        "labels": labels
    }