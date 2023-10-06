import numpy as np
import torch
import random

import PIL
import os
import torchvision
from torchvision import transforms
from ignite.metrics import PSNR
import matplotlib as mpl
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

from models.architectures import srflow
from utils import metrics, wasserstein
from geomloss import SamplesLoss
from operator import add
from scipy import ndimage
parser = argparse.ArgumentParser()

# train configs
parser.add_argument("--model", type=str, default="srflow",
                    help="Model you want to train.")
parser.add_argument("--modeltype", type=str, default="srflow",
                    help="Specify modeltype you would like to train [flow, unet3d].")
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
parser.add_argument("--datadir", type=str, default="data",
                    help="Dataset to train the model on.")
parser.add_argument("--trainset", type=str, default="era5-TCW",
                    help="Dataset to train the model on.")
parser.add_argument("--testset", type=str, default="era5-TCW",
                    help="Specify test dataset")

args = parser.parse_args()

def inv_scaler_temp(x):
    max_value = 315.91873
    min_value = 241.22385
    return x * (max_value - min_value) + min_value

def inv_scaler(x, ref=None):
    min_value = 0
    max_value = 100 
    return x * (max_value - min_value) + min_value

def plot_std(model, test_loader, exp_name, modelname, args):
    """
    For this experiment we visualize the super-resolution space for a single
    low-resolution image and its possible HR target predictions. We visualize
    the standard deviation of these predictions from the mean of the model.
    """
    color = 'plasma'
    savedir_viz = "experiments/{}_{}_{}/snapshots/population_std/".format(exp_name, modelname, args.trainset)
    os.makedirs(savedir_viz, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            y = item[0].to(args.device)
            x = item[1].to(args.device)

            y_unnorm = item[2].squeeze(1).to(args.device)
            x_unnorm = item[3].squeeze(1).to(args.device)

            mu0,_,_ = model(xlr=x, reverse=True, eps=0.0000000000000000001)

            samples = []
            n = 20
            sq_diff = torch.zeros_like(mu0)
            for n in range(n):
                mu1, _, _ = model(xlr=x, reverse=True, eps=0.8)
                samples.append(mu0)
                sq_diff += (mu1 - mu0)**2

            # compute population standard deviation
            sigma = torch.sqrt(sq_diff / n)

            # create plot
            plt.figure()
            plt.imshow(sigma[0,...].permute(2,1,0).cpu().numpy(), cmap=color)
            plt.axis('off')
            # plt.show()
            plt.savefig(savedir_viz + '/sigma_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.imshow(mu0[0,...].permute(2,1,0).cpu().numpy(), cmap='viridis')
            plt.axis('off')
            # plt.show()
            plt.savefig(savedir_viz + '/mu0_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1,7)
            # fig.suptitle('Y, Y_hat, mu, sigma')
            ax1.imshow(y[0,...].permute(2,1,0).cpu().numpy(), cmap='viridis')
            ax1.set_title('Ground Truth', fontsize=5)
            ax1.axis('off')
            ax2.imshow(mu0[0,...].permute(2,1,0).cpu().numpy(), cmap='viridis')
            ax2.set_title('Mean', fontsize=5)
            ax2.axis('off')
            ax3.imshow(samples[1][0,...].permute(2,1,0).cpu().numpy(), cmap='viridis')
            ax3.set_title('Sample 1', fontsize=5)
            ax3.axis('off')
            ax4.imshow(samples[2][0,...].permute(2,1,0).cpu().numpy(), cmap='viridis')
            ax4.set_title('Sample 2', fontsize=5)
            ax4.axis('off')
            ax5.imshow(samples[2][0,...].permute(2,1,0).cpu().numpy(), cmap='viridis')
            ax5.set_title('Sample 3', fontsize=5)
            ax5.axis('off')
            ax6.imshow(samples[2][0,...].permute(2,1,0).cpu().numpy(), cmap='viridis')
            ax6.set_title('Sample 4', fontsize=5)
            ax6.axis('off')
            ax7.imshow(sigma[0,...].permute(2,1,0).cpu().numpy(), cmap='magma')
            ax7.set_title('Std. Dev.', fontsize=5)
            ax7.axis('off')
            plt.tight_layout()
            plt.savefig(savedir_viz + '/std_multiplot_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()

    return None

def test(model, test_loader, exp_name, modelname, logstep, args):

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    nll_list=[]

    avrg_fwd_time = []
    avrg_bw_time = []

    # storing metrics
    ssim0 = []#  [0] * args.bsz
    ssim05 =[]# [0] * args.bsz
    ssim08 =[]# [0] * args.bsz
    ssim1 = []# [0] * args.bsz

    psnr0 = []# [0] * args.bsz
    psnr05 = []# [0] * args.bsz
    psnr08 =[]# [0] * args.bsz
    psnr1 = []# [0] * args.bsz

    mse0 = [0] * args.bsz
    mse05 = [0] * args.bsz
    mse08 = [0] * args.bsz
    mse1 = [0] * args.bsz

    mae0 = []
    mae05 = [] # [0] * args.bsz
    mae08 = []  # [0] * args.bsz
    mae1 = [] # [0] * args.bsz

    rmse0 = [0] * args.bsz
    rmse05 = [0] * args.bsz
    rmse08 = [0] * args.bsz
    rmse1 = [0] * args.bsz

    mmd0 = [0] * args.bsz
    mmd0 = [0] * args.bsz
    mmd05 = [0] * args.bsz
    mmd08 = [0] * args.bsz
    mmd1 = [0] * args.bsz

    emd = [0] * args.bsz
    rmse = [0] * args.bsz

    crps0 = [0] * args.bsz

    mae0_plot = []
    ssim0_dens = []
    psnr0_dens = []
    rmse0_dens = []

    color = 'inferno' if args.trainset == 'era5' else 'viridis'
    savedir_viz = "experiments/{}_{}_{}/snapshots/test/".format(exp_name, modelname, args.trainset)
    savedir_txt = 'experiments/{}_{}_{}/'.format(exp_name, modelname, args.trainset)

    os.makedirs(savedir_viz, exist_ok=True)
    os.makedirs(savedir_txt, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            y = item[0].to(args.device)
            x = item[1].to(args.device)

            y_unnorm = item[2].squeeze(1).to(args.device)
            x_unnorm = item[3].squeeze(1)

            z, nll = model.forward(x_hr=y, xlr=x)

            # Generative loss
            nll_list.append(nll.mean().detach().cpu().numpy())

            # evaluate for different temperatures
            mu0, _, _ = model(xlr=x, reverse=True, eps=0)
            mu05, _, _ = model(xlr=x, reverse=True, eps=0.5)
            mu08, _, _ = model(xlr=x, reverse=True, eps=0.8)
            mu1, _, _ = model(xlr=x, reverse=True, eps=1.0)

            # Visual Metrics
            print("Evaluate Predictions on visual metrics... ")

            # SSIM
            current_ssim_mu0 = metrics.ssim(inv_scaler(mu0,y_unnorm), y_unnorm)
            print('Current SSIM', current_ssim_mu0.item())
            ssim0.append(current_ssim_mu0.cpu().numpy())
            pd.Series(ssim0).hist()
            plt.xlabel('SSIM')
            plt.ylabel('nr samples')
            plt.savefig(savedir_viz + '/ssim0_density.png', dpi=300, bbox_inches='tight')
            plt.close()

            current_ssim_mu05 = metrics.ssim(inv_scaler(mu05,y_unnorm), y_unnorm)
            ssim05.append(current_ssim_mu05.cpu())

            current_ssim_mu08 = metrics.ssim(inv_scaler(mu08,y_unnorm), y_unnorm)
            ssim08.append(current_ssim_mu08.cpu())# = list(map(add, current_ssim_mu08, ssim08))

            current_ssim_mu1 = metrics.ssim(inv_scaler(mu1,y_unnorm), y_unnorm)
            ssim1.append(current_ssim_mu1.cpu()) # = list(map(add, current_ssim_mu1, ssim1))

            # PSNR
            current_psnr_mu0 = metrics.psnr(inv_scaler(mu0,y_unnorm), y_unnorm)
            # current_psnr_mu0 = metrics.psnr(mu0, y)
            psnr0.append(current_psnr_mu0) # list(map(add, current_psnr_mu0, psnr0))
            print('Current PSNR', current_psnr_mu0)
            # psnr0_dens.extend(current_psnr_mu0)
            pd.Series(psnr0).hist()
            plt.xlabel('PSNR')
            plt.ylabel('nr samples')
            plt.savefig(savedir_viz + '/psnr0_density.png', dpi=300, bbox_inches='tight')
            plt.close()

            current_psnr_mu05 = metrics.psnr(inv_scaler(mu05,y_unnorm), y_unnorm)
            # current_psnr_mu05 = metrics.psnr(mu05, y)
            psnr05.append(current_psnr_mu05) #= list(map(add, current_psnr_mu05, psnr05))

            # current_psnr_mu08 = metrics.psnr(mu08, y)
            current_psnr_mu08 = metrics.psnr(inv_scaler(mu08,y_unnorm), y_unnorm)
            psnr08.append(current_psnr_mu08) # = list(map(add, current_psnr_mu08, psnr08))

            # current_psnr_mu1 = metrics.psnr(mu1, y)
            current_psnr_mu1 = metrics.psnr(inv_scaler(mu1, y_unnorm), y_unnorm)
            psnr1.append(current_psnr_mu1) # = list(map(add, current_psnr_mu1, psnr1))

            # MSE
            current_mse0 = metrics.MSE(inv_scaler(mu0,y_unnorm), y_unnorm).detach().cpu().numpy()
            # current_mse0 = metrics.MSE(mu0, y).detach().cpu().numpy()*100
            mse0 = list(map(add, current_mse0, mse0))
            print('Current MSE', current_psnr_mu0.item())

            current_mse05 = metrics.MSE(mu05, y).detach().cpu().numpy()*100
            mse05 = list(map(add, current_mse05, mse05))

            current_mse08 = metrics.MSE(mu08,y).detach().cpu().numpy()*100
            mse08 = list(map(add, current_mse08, mse08))

            current_mse1 = metrics.MSE(mu1,y).detach().cpu().numpy()*100
            mse1 = list(map(add, current_mse1, mse1))

            # MAE
            current_mae0 = metrics.MAE(inv_scaler(mu0,y_unnorm),y_unnorm).detach().cpu().numpy()
            # current_mae0 = metrics.MAE(mu0,y).detach().cpu().numpy()*100
            mae0.append(current_mae0) # = list(map(add, current_mae0.cpu().numpy(), mae0))
            print('Current MAE', current_mae0.item())
            # mae0_plot.extend(current_mae0.detach().cpu().numpy().tolist())
            pd.Series(mae0).hist()
            plt.xlabel('MAE')
            plt.ylabel('nr samples')
            plt.savefig(savedir_viz + '/mae0_density.png', dpi=300, bbox_inches='tight')
            plt.close()

            current_mae05 = metrics.MAE(mu05,y).detach().cpu().numpy() * 100
            mae05.append(current_mae05) #list(map(add, current_mae05.cpu().numpy(), mae05))

            current_mae08 = metrics.MAE(mu08,y).detach().cpu().numpy() * 100
            mae08.append(current_mae08) #= list(map(add, current_mae08.cpu().numpy(), mae08))

            current_mae1 = metrics.MAE(mu1,y).detach().cpu().numpy() * 100
            mae1.append(current_mae1) #= list(map(add, current_mae1.cpu().numpy(), mae1))

            # RMSE
            # current_rmse0 = metrics.RMSE(inv_scaler(mu0,y_unnorm),y_unnorm)
            current_rmse0 = metrics.RMSE(mu0,y) * 100
            rmse0 = list(map(add, current_rmse0.cpu().numpy(), mse0))
            print('Current RMSE', current_rmse0[0].item())

            current_rmse05 = metrics.RMSE(mu05,y) * 100
            rmse05 = list(map(add, current_rmse05.cpu().numpy(), mse05))

            current_rmse08 = metrics.RMSE(mu08,y) * 100
            rmse08 = list(map(add, current_rmse08.cpu().numpy(), mse08))

            current_rmse1 = metrics.RMSE(mu1,y)*100
            rmse1 = list(map(add, current_rmse1.cpu().numpy(), mse1))

            # MMD
            current_mmd0 = metrics.MMD(inv_scaler(mu0,y_unnorm),y_unnorm)
            mmd0 = list(map(add, current_mmd0.cpu().numpy(), mmd0))

            current_mmd05 = metrics.MMD(inv_scaler(mu05,y_unnorm),y_unnorm)
            mmd05 = list(map(add, current_mmd05.cpu().numpy(), mmd05))

            current_mmd08 = metrics.MMD(inv_scaler(mu08,y_unnorm),y_unnorm)
            mmd08 = list(map(add, current_mmd08.cpu().numpy(), mmd08))

            current_mmd1 = metrics.MMD(inv_scaler(mu1,y_unnorm),y_unnorm)
            mmd1 = list(map(add, current_mmd1.cpu().numpy(), mse1))

            crps = []
            for i in range(8):
                currmu, _, _ = model(xlr=x, reverse=True, eps=1.0)
                crps.append(inv_scaler(currmu,y_unnorm))

            mu0crps = torch.stack(crps, dim=1)

            current_crps = metrics.crps_ensemble(y_unnorm, mu0crps)
            # crps0 = list(map(add, current_crps, crps0))
            crps0 += current_crps
            print('Current CRPS', current_crps[0])

            print('Visualize results ...')

            # Visualize low resolution GT
            grid_low_res = torchvision.utils.make_grid(x[0:9, :, :, :].cpu(), nrow=3)
            plt.figure()
            plt.imshow(grid_low_res.permute(1, 2, 0)[:,:,0], cmap=color)
            plt.axis('off')
            # plt.title("Low-Res GT (train)")
            # plt.show()
            plt.savefig(savedir_viz + '/low_res_gt{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            # Visualize High-Res GT
            grid_high_res_gt = torchvision.utils.make_grid(y[0:9, :, :, :].cpu(), nrow=3)
            plt.figure()
            plt.imshow(grid_high_res_gt.permute(1, 2, 0)[:,:,0], cmap=color)
            plt.axis('off')
            # plt.title("High-Res GT")
            # plt.show()
            plt.savefig(savedir_viz + '/high_res_gt_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            grid_mu0 = torchvision.utils.make_grid(mu0[0:9,:,:,:].cpu(), nrow=3)
            plt.figure()
            plt.imshow(grid_mu0.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
            plt.axis('off')
            # plt.title("Prediction at t (test), mu=0")
            plt.savefig(savedir_viz + "mu_0_logstep_{}_test.png".format(batch_idx), dpi=300,bbox_inches='tight')
            plt.close()

            grid_mu05 = torchvision.utils.make_grid(mu05[0:9,:,:,:].cpu(), nrow=3)
            plt.figure()
            plt.imshow(grid_mu0.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
            plt.axis('off')
            # plt.title("Prediction at t (test), mu=0.5")
            plt.savefig(savedir_viz + "mu_0.5_logstep_{}_test.png".format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            grid_mu08 = torchvision.utils.make_grid(mu08[0:9,:,:,:].cpu(), nrow=3)
            plt.figure()
            plt.imshow(grid_mu08.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
            plt.axis('off')
            # plt.title("Prediction at t (test), mu=0.8")
            plt.savefig(savedir_viz + "mu_0.8_logstep_{}_test.png".format(batch_idx), dpi=300,bbox_inches='tight')
            plt.close()

            grid_mu1 = torchvision.utils.make_grid(mu1[0:9,:,:,:].cpu(), nrow=3)
            plt.figure()
            plt.imshow(grid_mu1.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
            plt.axis('off')
            # plt.title("Prediction at t (test), mu=1.0")
            plt.savefig(savedir_viz + "mu_1_logstep_{}_test.png".format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            abs_err = torch.abs(mu08 - y)
            grid_abs_error = torchvision.utils.make_grid(abs_err[0:9,:,:,:].cpu(), nrow=3)
            plt.figure()
            plt.imshow(grid_abs_error.permute(1, 2, 0)[:,:,0], cmap=color)
            plt.axis('off')
            # plt.title("Abs Err")
            plt.savefig(savedir_viz + '/abs_err_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()

            # write results to file:
            with open(savedir_txt + '/nll_runtimes.txt'.format(exp_name, modelname),'w') as f:
                f.write('Avrg NLL: %d \n'% np.mean(nll_list))
                f.write('Avrg fwd. runtime: %.2f \n'% np.mean(avrg_fwd_time))
                f.write('Avrg bw runtime: %.2f'% np.mean(avrg_bw_time))

    # compute average metric values over test set
    avrg_ssim0 = np.mean(ssim0) # list(map(lambda x: x/len(test_loader), ssim0))
    avrg_ssim05 = np.mean(ssim05) # list(map(lambda x: x/len(test_loader), ssim05))
    avrg_ssim08 = np.mean(ssim08) # list(map(lambda x: x/len(test_loader), ssim08))
    avrg_ssim1 = np.mean(ssim1) # list(map(lambda x: x/len(test_loader), ssim1))

    avrg_psnr0 = np.mean(psnr0) #list(map(lambda x: x/len(test_loader), psnr0))
    avrg_psnr05 = np.mean(psnr05) #list(map(lambda x: x/len(test_loader), psnr05))
    avrg_psnr08 = np.mean(psnr08) # list(map(lambda x: x/len(test_loader), psnr08))
    avrg_psnr1 = np.mean(psnr1) # list(map(lambda x: x/len(test_loader), psnr1))

    avrg_mse0 = list(map(lambda x: x/len(test_loader), mse0))
    avrg_mse05 = list(map(lambda x: x/len(test_loader), mse05))
    avrg_mse08 = list(map(lambda x: x/len(test_loader), mse08))
    avrg_mse1 = list(map(lambda x: x/len(test_loader), mse1))

    avrg_mae0 = np.mean(mae0) # list(map(lambda x: x/len(test_loader), mae0))
    avrg_mae05 = np.mean(mae05) # list(map(lambda x: x/len(test_loader), mae05))
    avrg_mae08 = np.mean(mae08)# list(map(lambda x: x/len(test_loader), mae08))
    avrg_mae1 = np.mean(mae1) #list(map(lambda x: x/len(test_loader), mae1))

    avrg_rmse0 = list(map(lambda x: x/len(test_loader), rmse0))
    avrg_rmse05 = list(map(lambda x: x/len(test_loader), rmse05))
    avrg_rmse08 = list(map(lambda x: x/len(test_loader), rmse08))
    avrg_rmse1 = list(map(lambda x: x/len(test_loader), rmse1))

    avrg_mmd0 = list(map(lambda x: x/len(test_loader), mmd0))
    avrg_mmd05 = list(map(lambda x: x/len(test_loader), mmd05))
    avrg_mmd08 = list(map(lambda x: x/len(test_loader), mmd08))
    avrg_mmd1 = list(map(lambda x: x/len(test_loader),mmd1))

    avrg_crps = list(map(lambda x: x/len(test_loader), crps0))

    # Write metric results to a file in case to recreate plots
    with open(savedir_txt + 'metric_results.txt','w') as f:


        f.write('Avrg SSIM mu0:\n')
        f.write("%f \n" % np.mean(avrg_ssim0))

        f.write('Avrg SSIM mu05:\n')
        f.write("%f \n" %np.mean(avrg_ssim05))

        f.write('Avrg SSIM mu08:\n')
        f.write("%f \n" %np.mean(avrg_ssim08))

        f.write('Avrg SSIM mu1:\n')
        f.write("%f \n" %np.mean(avrg_ssim1))


        f.write('Avrg PSNR mu0:\n')
        f.write("%f \n" % np.mean(avrg_psnr0))

        f.write('Avrg PSNR mu05:\n')
        f.write("%f \n" %np.mean(avrg_psnr05))

        f.write('Avrg PSNR mu08:\n')
        f.write("%f \n" %np.mean(avrg_psnr08))

        f.write('Avrg PSNR mu1:\n')
        f.write("%f \n" %np.mean(avrg_psnr1))


        f.write('Avrg MSE mu0:\n')
        f.write("%f \n" %np.mean(avrg_mse0))

        f.write('Avrg MSE mu05:\n')
        f.write("%f \n" %np.mean(avrg_mse05))

        f.write('Avrg MSE mu08:\n')
        f.write("%f \n" %np.mean(avrg_mse08))

        f.write('Avrg MSE mu1:\n')
        f.write("%f \n" %np.mean(avrg_mse1))


        f.write('Avrg MAE mu0:\n')
        f.write("%f \n" %np.mean(avrg_mae0))

        f.write('Avrg MAE mu05:\n')
        f.write("%f \n" %np.mean(avrg_mae05))

        f.write('Avrg MAE mu08:\n')
        f.write("%f \n" %np.mean(avrg_mae08))

        f.write('Avrg MAE mu1:\n')
        f.write("%f \n" %np.mean(avrg_mae1))


        f.write('Avrg RMSE mu0:\n')
        f.write("%f \n" %np.mean(avrg_rmse0))

        f.write('Avrg RMSE mu05:\n')
        f.write("%f \n" %np.mean(avrg_rmse05))

        f.write('Avrg RMSE mu08:\n')
        f.write("%f \n" %np.mean(avrg_rmse08))

        f.write('Avrg RMSE mu1:\n')
        f.write("%f \n" %np.mean(avrg_rmse1))


        f.write('Avrg MMD mu0:\n')
        f.write("%f \n" %np.mean(avrg_mmd0))

        f.write('Avrg MMD mu05:\n')
        f.write("%f \n" %np.mean(avrg_mmd05))

        f.write('Avrg MMD mu08:\n')
        f.write("%f \n" %np.mean(avrg_mmd08))

        f.write('Avrg MMD mu1:\n')
        f.write("%f \n" %np.mean(avrg_mmd1))

        f.write('Avrg CRPS mu0:\n')
        f.write("%f \n" %np.mean(avrg_crps))

    print("Average Test Neg. Log Probability Mass:", np.mean(nll_list))
    print("Average Fwd. runtime", np.mean(avrg_fwd_time))
    print("Average Bw runtime:", np.mean(avrg_bw_time))

    return None

def calibration_exp(model, test_loader, exp_name, modelname, logstep, args):
    """
    For this experiment we visualize the pixel distribution of the normalized
    and unnormalized counterpart images of the ground truth and the predicted
    super-resolved image to assess the calibration.
    """

    savedir_viz = "experiments/{}_{}_{}/snapshots/calibration_histograms/".format(exp_name, modelname, args.trainset)
    os.makedirs(savedir_viz, exist_ok=True)

    print("Generating Histograms ... ")

    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            y = item[0].to(args.device)
            x = item[1].to(args.device)

            y_unorm = item[2].squeeze(1).to(args.device)
            x_unorm = item[3].squeeze(1).to(args.device)

            # import pdb; pdb.set_trace()

            # super resolve image
            mu05, _, _ = model(xlr=x, reverse=True, eps=0.5)

            n_bins = 100
            fig, ((ax0,ax1)) = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))

            colors = ['mediumorchid', 'coral']

            labelax0 = [r'$\tilde{y}$',r'$\tilde{\hat{y}}$']
            y_fl = y.flatten().detach().cpu().numpy()
            mu05_fl = mu05.flatten().detach().cpu().numpy()
            ax0.hist(np.stack((y_fl, mu05_fl),axis=1), n_bins, density=True, histtype='step',color=colors, label=labelax0)
            ax0.set_xlabel('nr of bins')
            ax0.set_ylabel('pixel values')
            ax0.set_title('Normalized prediction vs. ground truth pixel distribution')
            ax0.legend(prop={'size': 10})

            labelax1 = ['y',r'$\hat{y}$']
            mu05_unorm = inv_scaler(mu05, ref=y_unorm)
            y_unorm_fl = y_unorm.flatten().detach().cpu().numpy()
            mu05_unorm_fl = mu05_unorm.flatten().detach().cpu().numpy()
            ax1.hist(np.stack((y_unorm_fl, mu05_unorm_fl),axis=1), n_bins, density=True, histtype='step',color=colors, label=labelax1)
            ax1.set_xlabel('nr of bins')
            ax1.set_ylabel('pixel values')
            ax1.set_title('Unormalized prediction vs. ground truth pixel distribution')
            ax1.legend(prop={'size': 10})

            # plt.show()

            # labelax2 = ['abs diff normalized']
            # newcolor = ['deepskyblue', 'palegreen']
            # diff = torch.abs(y-mu05).flatten().detach().cpu().numpy()
            # diff_unorm = torch.abs(y_unorm-mu05_unorm).flatten().detach().cpu().numpy()
            # ax2.hist(np.stack((diff, diff_unorm),axis=1), n_bins, density=True, histtype='barstacked',color=newcolor, label=labelax2)
            # ax2.set_xlabel('nr of bins')
            # ax2.set_ylabel('pixel values')
            # ax2.set_title('Absolute difference distributions')
            # ax2.legend(prop={'size': 10})
            # plt.show()

            fig.savefig(savedir_viz + '/histogram_multiplot_{}.png'.format(batch_idx), dpi=300, bbox_inches='tight')
            plt.close()



if __name__ == "__main__":

    print(torch.cuda.device_count())

    # NOTE: when executing code, make sure you enable the --testmode flag !!!

    # Load testset
    _, _, test_loader, args = dataloading.load_data(args)

    in_channels = next(iter(test_loader))[0].shape[1]
    args.height, args.width = next(iter(test_loader))[0].shape[2], next(iter(test_loader))[0].shape[3]

    args.device = "cuda"

    # Load Model
    # temperature
    # modelname = 'model_epoch_35_step_23750'
    # modelpath = '/home/christina/Documents/clim-var-ds-cnf/runs/srflow_era5_2023_09_08_14_13_03/model_checkpoints/{}.tar'.format(modelname)
    # watercontent 4x upsampling

    # watercontent 2x upsampling
    modelname = 'model_epoch_0_step_250'
    modelpath = '/home/christina/Documents/clim-var-ds-cnf/runs/srflow_era5-TCW_2023_10_06_12_42_11/model_checkpoints/{}.tar'.format(modelname)

    # 4x upsampling
    # modelname = 'model_epoch_2_step_27000'
    # modelpath = '/home/christina/Documents/clim-var-ds-cnf/runs/srflow_era5-TCW_2023_09_28_22_36_08_4x/model_checkpoints/{}.tar'.format(modelname)

    model = srflow.SRFlow((in_channels, args.height, args.width), args.filter_size, args.L, args.K,
                           args.bsz, args.s, args.nb, args.condch, args.nbits, args.noscale, args.noscaletest)

    print(torch.cuda.device_count())
    ckpt = torch.load(modelpath, map_location='cuda:0')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params {}:  '.format(args.device), params)
    model = model.to(args.device)

    exp_name = "flow-{}-level-{}-k".format(args.L, args.K)
    # plot_std(model, test_loader, exp_name, modelname, args)
    calibration_exp(model, test_loader, exp_name, modelname, -99999, args)

    print("Evaluate on test split ...")
    # test(model, test_loader, exp_name, modelname, -99999, args)
