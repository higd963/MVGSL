import torch
from torch.utils.data import Dataset
import numpy as np

class WeatherDiseaseDataset(Dataset):
    def __init__(self, inputx_path, inputy_path, split='train'):
        self.inputx = torch.tensor(np.load(inputx_path), dtype=torch.float32)
        self.inputy = torch.tensor(np.load(inputy_path), dtype=torch.float32)

        total_len = len(self.inputx)
        train_len = int(total_len * 3 / 5)
        val_len = int(total_len * 1 / 5)
        test_len = total_len - train_len - val_len

        if split == 'train':
            self.inputx = self.inputx[:train_len]
            self.inputy = self.inputy[:train_len]
        elif split == 'val':
            self.inputx = self.inputx[train_len:train_len + val_len]
            self.inputy = self.inputy[train_len:train_len + val_len]
        elif split == 'test':
            self.inputx = self.inputx[train_len + val_len:]
            self.inputy = self.inputy[train_len + val_len:]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.inputx)

    def __getitem__(self, idx):
        return self.inputx[idx], self.inputy[idx]

