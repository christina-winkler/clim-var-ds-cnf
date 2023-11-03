import numpy as np
import torch
import random

import PIL
import os
import torchvision
from torchvision import transforms
from ignite.metrics import PSNR
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import sys
sys.path.append("../../")

# seeding only for debugging
# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Dataset loading
from data import dataloading
from data.era5_temp_dataset import InverseMinMaxScaler

from os.path import exists, join
import matplotlib.pyplot as plt
from matplotlib import transforms
import argparse
import timeit
import pdb
import seaborn as sns

from models.architectures import cdiff
from utils import metrics, wasserstein
from geomloss import SamplesLoss
from operator import add
from scipy import ndimage
parser = argparse.ArgumentParser()

# train configs
parser.add_argument("--modeltype", type=str, default="srgan",
                    help="Specify modeltype you would like to train [srflow, cdiff, stflow].")
parser.add_argument("--model_path", type=str, default="runs/",
                    help="Directory where models are saved.")
parser.add_argument("--modelname", type=str, default=None,
                    help="Sepcify modelname to be tested.")
parser.add_argument("--epochs", type=int, default=20000,
                    help="number of epochs")
parser.add_argument("--max_steps", type=int, default=2000000,
                    help="For training on a large dataset.")
parser.add_argument("--log_interval", type=int, default=250,
                    help="Interval in which results should be logged.")
parser.add_argument("--val_interval", type=int, default=250,
                    help="Interval in which model should be validated.")

# runtime configs
parser.add_argument("--visual", action="store_true",
                    help="Visualizing the samples at test time.")
parser.add_argument("--noscaletest", action="store_true",
                    help="Disable scale in coupling layers only at test time.")
parser.add_argument("--noscale", action="store_true",
                    help="Disable scale in coupling layers.")
parser.add_argument("--testmode", action="store_true",
                    help="Model run on test set.")
parser.add_argument("--train", action="store_true",
                    help="If model should be trained.")
parser.add_argument("--resume", action="store_true",
                    help="If training should be resumed.")

# hyperparameters
parser.add_argument("--nbits", type=int, default=8,
                    help="Images converted to n-bit representations.")
parser.add_argument("--s", type=int, default=2, help="Upscaling factor.")
parser.add_argument("--gauss_steps", type=int, default=1000,
                    help="Number of gaussianization steps in diffusion process.")
parser.add_argument("--noise_sched", type=str, default='cosine',
                    help="Type of noise schedule defining variance of noise that is added to the data in the diffusion process.")
parser.add_argument("--crop_size", type=int, default=500,
                    help="Crop size when random cropping is applied.")
parser.add_argument("--patch_size", type=int, default=500,
                    help="Training patch size.")
parser.add_argument("--bsz", type=int, default=16, help="batch size")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="learning rate")
parser.add_argument("--filter_size", type=int, default=512//2,
                    help="filter size NN in Affine Coupling Layer")
parser.add_argument("--L", type=int, default=3, help="# of levels")
parser.add_argument("--K", type=int, default=2,
                    help="# of flow steps, i.e. model depth")
parser.add_argument("--nb", type=int, default=16,
                    help="# of residual-in-residual blocks LR network.")
parser.add_argument("--condch", type=int, default=128//8,
                    help="# of residual-in-residual blocks in LR network.")
parser.add_argument("--linear_start", type=float, default=1e-6,
                    help="Minimum value of the linear schedule (for diffusion model).")
parser.add_argument("--linear_end", type=float, default=1e-2,
                    help="Maximum value of the linear schedule (for diffusion model).")

# data
parser.add_argument("--datadir", type=str, default="data",
                    help="Dataset to train the model on.")
parser.add_argument("--trainset", type=str, default="era5-TCW",
                    help="Dataset to train the model on.")
parser.add_argument("--vminmax", type=int, default=(0,100),
                    help="Values according to which the plots are normalized.")
args = parser.parse_args()

if torch.cuda.is_available():
    args.device = torch.device("cuda")
    args.num_gpus = torch.cuda.device_count()
    args.parallel = False

else:
    args.device = "cpu"

def inv_scaler(x, min_value=0, max_value=100):
    # min_value = 0 if args.trainset == 'era5-TCW' else 315.91873
    # max_value = 100 if args.trainset == 'era5-TCW' else 241.22385
    x = x * (max_value - min_value) + min_value
    # return
    return x

if __name__ == "__main__":

    # Load testset
    _, _, test_loader, args = dataloading.load_data(args)

    in_channels = next(iter(test_loader))[0].shape[1]
    height, width = next(iter(test_loader))[0].shape[2], next(iter(test_loader))[0].shape[3]

    args.device = "cuda"

    # Load Model
    modelname = 'model_epoch_0_step_5'
    modelpath = '/home/christina/Documents/clim-var-ds-cnf/runs/cdiff_era5-TCW_2023_10_26_15_22_22/model_checkpoints/{}.tar'.format(modelname)

    model = cdiff.CondDiffusion(input_shape=(in_channels, height, width),
                                bsz=args.bsz, s=args.s, nb=args.nb, cond_channels=args.condch,
                                trainmode=args.train, device=args.device,
                                conditional=True, linear_start=1e-6, linear_end=1e-2,
                                noise_sched=args.noise_sched, T=args.gauss_steps)

    print(torch.cuda.device_count())
    ckpt = torch.load(modelpath, map_location='cuda:0')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params {}:  '.format(args.device), params)
    model = model.to(args.device)
