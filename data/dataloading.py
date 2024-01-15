from os.path import exists, join
from os import listdir
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
import torch
import random
import sys
import pdb
import os
import xarray as xr

from data.era5_temp_dataset import ERA5T2MData
from data.weatherbench_dataset import WeatherBenchData

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# import cv2
import os
sys.path.append("../")

from data.imresize_bicubic import imresize

# data utils

def load_era5(args):

    print("Loading ERA5 ...")

    dpath = os.getcwd() + '/data/assets/ftp.bgc-jena.mpg.de/pub/outgoing/aschall/data.zarr'
    dataset = ERA5T2MData(data_path=dpath, window_size=args.lag_len)

    n_train_samples = int(len(dataset) // (1/0.7))
    n_val_samples = int(len(dataset) // (1/0.2))
    n_test_samples = int(len(dataset) // (1/0.1))

    train_idcs = [i for i in range(0, n_train_samples)]
    val_idcs = [i for i in range(0, n_val_samples)]
    test_idcs = [i for i in range(0, n_test_samples)]

    trainset = torch.utils.data.Subset(dataset, train_idcs)
    valset = torch.utils.data.Subset(dataset, val_idcs)
    testset = torch.utils.data.Subset(dataset, test_idcs)

    train_loader = data_utils.DataLoader(trainset, args.bsz, shuffle=True,
                                         drop_last=True)
    val_loader = data_utils.DataLoader(valset, args.bsz, shuffle=True,
                                       drop_last=True)
    test_loader = data_utils.DataLoader(testset, args.bsz, shuffle=False,
                                        drop_last=True)

    return train_loader, val_loader, test_loader, args

def load_weather_bench(args):

    print("Loading Weather Bench ...")

    dpath = args.datadir + '/geopotential_500_orig/'
    dataset = WeatherBenchData(data_path=dpath, window_size=2, args=args)

    n_train_samples = int(len(dataset) // (1/0.85))
    n_val_samples = int(len(dataset) // (1/0.05))
    n_test_samples = int(len(dataset) // (1/0.1))

    train_idcs = [i for i in range(0, n_train_samples)]
    val_idcs = [i for i in range(0, n_val_samples)]
    test_idcs = [i for i in range(0, n_test_samples)]

    trainset = torch.utils.data.Subset(dataset, train_idcs)
    valset = torch.utils.data.Subset(dataset, val_idcs)
    testset = torch.utils.data.Subset(dataset, test_idcs)

    train_loader = data_utils.DataLoader(trainset, args.bsz, shuffle=True,
                                         drop_last=True)
    val_loader = data_utils.DataLoader(valset, args.bsz, shuffle=True,
                                       drop_last=True)
    test_loader = data_utils.DataLoader(testset, args.bsz, shuffle=False,
                                        drop_last=True)

    return train_loader, val_loader, test_loader, args


def load_data(args):

    if args.trainset == "era5":
        return load_era5(args)

    elif args.trainset == "wbench":
        return load_weather_bench(args)

    else:
        raise ValueError("Dataset not available. Check for typos!")
