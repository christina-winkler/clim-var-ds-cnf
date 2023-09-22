from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from torchvision import transforms
from dataclasses import dataclass
from typing import Tuple, Callable
from skimage.transform import resize
import torch
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class MinMaxScaler:
    max_value: float
    min_value: float
    values_range: Tuple[int, int] = (-1, 1)

    def __call__(self, x):
        x = (x - self.min_value) / (self.max_value - self.min_value)
        return x * (self.values_range[1] - self.values_range[0]) + self.values_range[0]

@dataclass
class InverseMinMaxScaler:
    max_value: float
    min_value: float
    values_range: Tuple[int, int] = (0, 1)

    def __call__(self, y):
        x = y * (self.max_value - self.min_value) + self.min_value
        return x


@dataclass
class ERA5WTCData(Dataset):
    """
    """

    data_path: str
    testmode: False
    s: 4

    def __post_init__(self):

        self.inputs = torch.load(self.data_path + '/input.pt')
        self.targets = torch.load(self.data_path + '/target.pt')

        self.mean = self.targets.mean()
        self.std = self.targets.std()

        min_val = self.targets.min()
        max_val = self.targets.max()

        self.transform = MinMaxScaler(max_value=max_val, min_value=min_val,
                                      values_range=(0,1))
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x,y = self.inputs[idx], self.targets[idx]

        if self.s==2:
            # import pdb; pdb.set_trace()
            x = np.zeros((1,1,y.shape[2]//2,y.shape[2]//2))
            x[0,0,...] = resize(y[0,0,...], (y.shape[2]//2, y.shape[3]//2), anti_aliasing=True)
            transf = transforms.ToTensor()
            x = transf(x[0,...]).permute(1,2,0).unsqueeze(0).type(torch.FloatTensor)

        if not self.testmode:
            return self.transform(y).squeeze(1), self.transform(x).squeeze(1)
        else:
            return self.transform(y).squeeze(1), self.transform(x).squeeze(1), y,x
