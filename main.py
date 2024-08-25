import sys
sys.path.append("../../")

import argparse
import torch
import random
import numpy as np
import os

# Dataset loading
from data import dataloading

# Models
from models.architectures import srflow, srgan, cdiffusion

# Optimization
from optimization import trainer_srflow, trainer_srgan, trainer_cdiff

from utils import *

# import evaluate
import test
import pdb
from tensorboardX import SummaryWriter

import sys
sys.path.append("../../")

###############################################################################


def main(args):

    print('Num of avail GPUs:', torch.cuda.device_count())

    # Initialize device on which to run the model
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.num_gpus = torch.cuda.device_count()
        args.parallel = False

    else:
        args.device = "cpu"

    # args.device = 'cpu'
    print("Device", args.device)
    # args.device = "cpu"

    # Build name of current model
    if args.modelname is None:
        args.modelname = "{}_{}_bsz{}_K{}_L{}_lr{:.4f}_s{}".format(args.modeltype, args.trainset,
                                                                   args.bsz, args.K, args.L,
                                                                   args.lr, args.s)

    if args.train:
        # load data
        train_loader, valid_loader, test_loader, args = dataloading.load_data(args)
        in_channels = next(iter(test_loader))[0].shape[1]
        height, width = next(iter(train_loader))[0].shape[2], next(iter(train_loader))[0].shape[3]

    print("Start training {} on {}:".format(args.modeltype, args.trainset))

    if args.modeltype == "srflow":

        print("Hard Constraint:", args.constraint)
        model = srflow.SRFlow((in_channels, height, width), args.filter_size, args.L, args.K,
                               args.bsz, args.s, args.constraint, args.nb, args.condch, args.noscale, args.noscaletest)
        if args.resume:
            modelname = 'model_epoch_1_step_53000.tar'
            modelpath = "/home/christina/Documents/clim-var-ds-cnf/runs/srflow_era5-TCW_2023_10_02_18_59_012x/model_checkpoints/{}".format(modelname)
            ckpt = torch.load(modelpath)
            model.load_state_dict(ckpt['model_state_dict'])

        trainer_srflow.trainer(args=args, train_loader=train_loader,
                               valid_loader=valid_loader,
                               model=model,
                               device=args.device)


    if args.modeltype == "srgan":

        generator = srgan.Generator(in_channels, out_nc=1, height=height, width=width, nf=128, s=args.s, nb=5)
        discriminator = srgan.Discriminator(in_channels, out_c=1, height=height, width=width)
        model = (generator, discriminator)

        if args.resume:
            modelname = 'model_epoch_1_step_53000.tar'
            modelpath = "/home/christina/Documents/clim-var-ds-cnf/runs/srflow_era5-TCW_2023_10_02_18_59_012x/model_checkpoints/{}".format(modelname)
            ckpt = torch.load(modelpath)
            model.load_state_dict(ckpt['model_state_dict'])

        trainer_srgan.trainer(args=args, train_loader=train_loader,
                              valid_loader=valid_loader,
                              model=model,
                              device=args.device)

        if args.resume:
            modelname = 'model_epoch_4_step_96500.tar'
            modelpath = os.getcwd() + "/home/christina/Documents/clim-var-ds-cnf/runs/srflow_era5-TCW_2023_09_22_17_08_51/model_checkpoints/{}".format(modelname)
            ckpt = torch.load(modelpath)
            model.load_state_dict(ckpt['model_state_dict'])

        trainer_stflow.trainer(args=args, train_loader=train_loader,
                               valid_loader=valid_loader,
                               model=model,
                               device=args.device)

    if args.modeltype == "cdiff":

        # model
        opt = utils.load_config("cdiff_config.json")
        model = cdiffusion.DDPM(opt)

        if args.resume:
            modelname = 'model_epoch_1_step_53000.tar'
            modelpath = "/home/christina/Documents/clim-var-ds-cnf/runs/srflow_era5-TCW_2023_10_02_18_59_012x/model_checkpoints/{}".format(modelname)
            ckpt = torch.load(modelpath)
            model.load_state_dict(ckpt['model_state_dict'])

        trainer_cdiff.trainer(args=args, train_loader=train_loader,
                              valid_loader=valid_loader,
                              model=model,
                              opt=opt,
                              device=args.device)

    else:
         print("Modeltype not available! Check spelling.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # train configs
    parser.add_argument("--modeltype", type=str, default="srflow",
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
    parser.add_argument("--addstring", type=str, default='',
                        help="String to be added to experiment name.")

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
    parser.add_argument("--s", type=int, default=2, help="Upscaling factor.")
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
    parser.add_argument("--constraint", type=str, default='None' ,help="type of constraint to apply to loss func: [None, add, softmax, scadd, mul]")

    # diffusion model hparams
    parser.add_argument("--linear_start", type=float, default=1e-6,
                        help="Minimum value of the linear schedule (for diffusion model).")
    parser.add_argument("--linear_end", type=float, default=1e-2,
                        help="Maximum value of the linear schedule (for diffusion model).")
    parser.add_argument("--gauss_steps", type=int, default=1000,
                        help="Number of gaussianization steps in diffusion process.")
    parser.add_argument("--noise_sched", type=str, default='cosine',
                        help="Type of noise schedule defining variance of noise that is added to the data in the diffusion process.")

    # data
    parser.add_argument("--datadir", type=str, default="/home/mila/c/christina.winkler/scratch/data/",
                        help="Path to data storage.")
    parser.add_argument("--trainset", type=str, default="era5-TCW",
                        help="Dataset to train the model on.")


    args = parser.parse_args()
    main(args)
