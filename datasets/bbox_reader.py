import torch
import numpy as np

from torch.utils.data import Dataset
from monai.transforms import apply_transform

class Luna16(Dataset):
    def  __init__(self, data : str, transform, max_stride=(4, 4, 4), spacing : tuple = (0.703125, 0.703125, 1.25), mode : str = 'train') -> None:
        self.data = data
        self.transform = transform
        self.stride = max_stride
        self.spacing = spacing
        self.mode = mode

    def _transform(self, idx):
        data_i = self.data[idx]
        return apply_transform(self.transform, data_i)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        targets = self._transform(index)      
        return targets
        
    

