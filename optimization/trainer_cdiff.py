from datetime import datetime
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.optim as optim
import os
import json
from skimage.transform import resize

# Utils
from utils import utils
import numpy as np
import random
import pdb
import torchvision
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from optimization.validation_cdiff import validate
from typing import Tuple, Callable

import wandb
os.environ["WANDB_SILENT"] = "true"
import sys
sys.path.append("../../")

# seeding only for debugging
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def trainer(args, train_loader, valid_loader, model,
            device='cpu', needs_init=True):

    cmap = 'viridis' if args.trainset == 'era5-TCW' else 'inferno'
    config_dict = vars(args)

    # write config dict to file


    # wandb.init(project="arflow", config=config_dict)
    args.experiment_dir = os.path.join('runs',
                                        args.modeltype + '_' + args.trainset  + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"))
    # set viz dir
    viz_dir = "{}/snapshots/trainset/".format(args.experiment_dir)
    os.makedirs(viz_dir, exist_ok=True)
