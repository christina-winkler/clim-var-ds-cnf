import numpy as np
import torch
import torch.nn as nn
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

from models.architectures import srgan, srgan2
from utils import metrics, wasserstein
from geomloss import SamplesLoss
from operator import add
from scipy import ndimage

parser = argparse.ArgumentParser()


# train configs
parser.add_argument("--model", type=str, default="srflow",
                    help="Model you want to train.")
parser.add_argument("--modeltype", type=str, default="srflow",
                    help="Specify modeltype you would like to train [srflow, cdiff, srgan].")
parser.add_argument("--model_path", type=str, default="runs/",
                    help="Directory where models are saved.")
parser.add_argument("--modelname", type=str, default=None,
                    help="Sepcify modelname to be tested.")
parser.add_argument("--epochs", type=int, default=10000,
                    help="number of epochs")
parser.add_argument("--max_steps", type=int, default=2000000,
                    help="For training on a large dataset.")
parser.add_argument("--log_interval", type=int, default=100,
                    help="Interval in which results should be logged.")

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
parser.add_argument("--resume_training", action="store_true",
                    help="If training should be resumed.")

# hyperparameters
parser.add_argument("--nbits", type=int, default=8,
                    help="Images converted to n-bit representations.")
parser.add_argument("--s", type=int, default=2, help="Upscaling factor.")
parser.add_argument("--crop_size", type=int, default=500,
                    help="Crop size when random cropping is applied.")
parser.add_argument("--patch_size", type=int, default=500,
                    help="Training patch size.")
parser.add_argument("--bsz", type=int, default=16, help="batch size")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="learning rate")
parser.add_argument("--filter_size", type=int, default=512,
                    help="filter size NN in Affine Coupling Layer")
parser.add_argument("--L", type=int, default=3, help="# of levels")
parser.add_argument("--K", type=int, default=2,
                    help="# of flow steps, i.e. model depth")
parser.add_argument("--nb", type=int, default=16,
                    help="# of residual-in-residual blocks LR network.")
parser.add_argument("--condch", type=int, default=128//8,
                    help="# of residual-in-residual blocks in LR network.")

# data
parser.add_argument("--datadir", type=str, default="/home/mila/c/christina.winkler/scratch/data",
                    help="Dataset to train the model on.")
parser.add_argument("--trainset", type=str, default="era5-TCW", help='[era5-TCW, era5-T2M]')
parser.add_argument("--testset", type=str, default="era5-TCW",
                    help="Specify test dataset")

args = parser.parse_args()


if torch.cuda.is_available():
    args.device = torch.device("cuda")
    args.num_gpus = torch.cuda.device_count()
    args.parallel = False

else:
    args.device = "cpu"


def inv_scaler(x, min_value=0, max_value=100):
    x = x * (max_value - min_value) + min_value
    return x

def plot_std(model):
    return None

def test(model, test_loader, exp_name, modelname, args):
    
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        nll_list=[]

        avrg_fwd_time = []
        avrg_bw_time = []

        mse = []
        mae = []
        rmse = [] 
        crps = [] 

        color = 'inferno' if args.trainset == 'era5-T2M' else 'viridis'
        savedir_viz = "experiments/{}_{}_{}/snapshots/test/".format(exp_name, modelname, args.trainset)
        savedir_txt = 'experiments/{}_{}_{}/'.format(exp_name, modelname, args.trainset)

        os.makedirs(savedir_viz, exist_ok=True)
        os.makedirs(savedir_txt, exist_ok=True)

        generator.eval()

        mse_loss_list = []
        mse_loss = nn.MSELoss()

        with torch.no_grad():
            for batch_idx, item in enumerate(test_loader):

                y = item[0].to(args.device)
                x = item[1].to(args.device)

                y_unorm = item[2].to(args.device)
                x_unorm = item[3].to(args.device)

                fake_img = generator(x)

                g_loss = mse_loss(fake_img, y)
                    
                # Generative loss
                mse_loss_list.append(g_loss.mean().detach().cpu().numpy())


                print("Evaluate Predictions on visual metrics... ")

                # MAE
                mae.append(metrics.MAE(inv_scaler(fake_img, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).detach().cpu().numpy())
                    
                print('Current MAE', np.mean(mae),mae[-1])

                mse.append(metrics.MSE(inv_scaler(fake_img, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).mean().detach().cpu().numpy())
                print('Current MSE', np.mean(mse), mse[-1])
            
                # RMSE
                rmse.append(metrics.RMSE(inv_scaler(fake_img, min_value=y_unorm.min(), max_value=y_unorm.max()), y_unorm).detach().cpu().numpy())
                print('Current RMSE', np.mean(rmse), rmse[-1])

                # CRPS
                imgs = []
                for i in range(8):
                    img = generator(x)
                    imgs.append(inv_scaler(img, min_value=y_unorm.min(), max_value=y_unorm.max()))

                crps_stack = torch.stack(imgs, dim=1)

                crps.append(metrics.crps_ensemble(y_unorm, crps_stack))

                print('Current CRPS', np.mean(crps))

                print('Visualize results ...')

                # Visualize low resolution GT
                grid_low_res = torchvision.utils.make_grid(x[0:9, :, :, :].cpu(), normalize=True, nrow=3)
                plt.figure()
                plt.imshow(grid_low_res.permute(1, 2, 0)[:,:,0], cmap=color)
                plt.axis('off')
                # plt.title("Low-Res GT (train)")
                # plt.show()
                plt.savefig(savedir_viz + '/low_res_gt{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
                plt.close()

                # Visualize High-Res GT
                grid_high_res_gt = torchvision.utils.make_grid(y[0:9, :, :, :].cpu(), normalize=True, nrow=3)
                plt.figure()
                plt.imshow(grid_high_res_gt.permute(1, 2, 0)[:,:,0], cmap=color)
                plt.axis('off')
                # plt.title("High-Res GT")
                # plt.show()
                plt.savefig(savedir_viz + '/high_res_gt_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
                plt.close()

                grid_pred = torchvision.utils.make_grid(fake_img[0:9,:,:,:].cpu(), normalize=True, nrow=3)
                plt.figure()
                plt.imshow(grid_pred.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                plt.axis('off')
                # plt.title("Prediction (test),")
                plt.savefig(savedir_viz + "y_hat_logstep_{}_test.png".format(batch_idx), dpi=300,bbox_inches='tight')
                plt.close()

                abs_err = torch.abs(fake_img - y)
                grid_abs_error = torchvision.utils.make_grid(abs_err[0:9,:,:,:].cpu(), normalize=True, nrow=3)
                plt.figure()
                plt.imshow(grid_abs_error.permute(1, 2, 0)[:,:,0], cmap=color)
                plt.axis('off')
                # plt.title("Abs Err")
                plt.savefig(savedir_viz + '/abs_err_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
                plt.close()

        # Write metric results to a file in case to recreate plots
        with open(savedir_txt + 'metric_results.txt','w') as f:
            f.write('MAE:\n')
            f.write("%f \n" %np.mean(mae))
            f.write("%f \n" %np.std(mae))

            f.write('RMSE:\n')
            f.write("%f \n" %np.mean(rmse))
            f.write("%f \n" %np.std(rmse))

            f.write('CRPS:\n')
            f.write("%f \n" %np.mean(crps))
            f.write("%f \n" %np.std(crps))
        
        return None

if __name__ == "__main__":

    # Load testset
    _, _, test_loader, args = dataloading.load_data(args)
    in_channels = next(iter(test_loader))[0].shape[1]
    args.height, args.width = next(iter(test_loader))[0].shape[2], next(iter(test_loader))[0].shape[3]

    # init model
    generator = srgan2.RRDBNet(in_channels, out_nc=1, nf=128, s=args.s, nb=5)
    # disc_net = srgan.Discriminator(in_channels)

    # Load model
    # 2x watercontent
    # modelname = 'generator_epoch_3_step_9500'
    # gen_modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srgan_era5-TCW_2023_11_08_11_16_25_2x/model_checkpoints/{}.tar'.format(modelname)

    # 4x watercontent
    modelname = 'generator_epoch_6_step_4000'
    gen_modelpath = '/home/mila/c/christina.winkler/clim-var-ds-cnf/runs/srgan_era5-TCW_2023_11_09_06_49_00_4x/model_checkpoints/{}.tar'.format(modelname)

    ckpt = torch.load(gen_modelpath)
    generator.load_state_dict(ckpt['model_state_dict'])
    generator.eval()

    params = sum(x.numel() for x in generator.parameters() if x.requires_grad)
    print('Nr of Trainable Params {}:  '.format(args.device), params)
    generator = generator.to(args.device)

    exp_name = "srgan-{}-{}x".format(args.trainset, args.s)
    test(generator, test_loader, exp_name, modelname, args)

