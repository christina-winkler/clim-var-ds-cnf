import numpy as np
import torch
import random

import PIL
import os
import torchvision
from torchvision import transforms
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
    min_value = ref.min()
    max_value = ref.max()
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
    ssim0 = [0] * args.bsz
    ssim05 = [0] * args.bsz
    ssim08 = [0] * args.bsz
    ssim1 = [0] * args.bsz

    psnr0 = [0] * args.bsz
    psnr05 = [0] * args.bsz
    psnr08 = [0] * args.bsz
    psnr1 = [0] * args.bsz

    mse0 = [0] * args.bsz
    mse05 = [0] * args.bsz
    mse08 = [0] * args.bsz
    mse1 = [0] * args.bsz

    mae0 = [0] * args.bsz
    mae05 = [0] * args.bsz
    mae08 = [0] * args.bsz
    mae1 = [0] * args.bsz

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

            start = timeit.default_timer()
            z, nll = model.forward(x_hr=y, xlr=x)
            stop = timeit.default_timer()
            print("Time Fwd pass:", stop-start)
            avrg_fwd_time.append(stop-start)

            # Generative loss
            nll_list.append(nll.mean().detach().cpu().numpy())

            # evalutae for different temperatures
            # import pdb; pdb.set_trace()
            mu0, _, _ = model(xlr=x, reverse=True, eps=0.2)
            mu05, _, _ = model(xlr=x, reverse=True, eps=0.5)
            mu08, _, _ = model(xlr=x, reverse=True, eps=0.8)
            mu1, _, _ = model(xlr=x, reverse=True, eps=1.0)

            # Visual Metrics
            print("Evaluate Predictions on visual metrics... ")

            # import pdb; pdb.set_trace()

            # SSIM
            current_ssim_mu0 = metrics.ssim(inv_scaler(mu0,y_unnorm), y_unnorm)
            print('Current SSIM', current_ssim_mu0[0])
            ssim0 = list(map(add, current_ssim_mu0, ssim0))
            ssim0_dens.extend(current_ssim_mu0)
            pd.Series(ssim0_dens).hist()
            plt.xlabel('SSIM')
            plt.ylabel('nr samples')
            plt.savefig(savedir_viz + '/ssim0_density.png', dpi=300, bbox_inches='tight')
            plt.close()

            current_ssim_mu05 = metrics.ssim(inv_scaler(mu05,y_unnorm), y_unnorm)
            ssim05 = list(map(add, current_ssim_mu05, ssim05))

            current_ssim_mu08 = metrics.ssim(inv_scaler(mu08,y_unnorm), y_unnorm)
            ssim08 = list(map(add, current_ssim_mu08, ssim08))

            current_ssim_mu1 = metrics.ssim(inv_scaler(mu1,y_unnorm), y_unnorm)
            ssim1= list(map(add, current_ssim_mu1, ssim1))

            # PSNR
            current_psnr_mu0 = metrics.psnr(inv_scaler(mu0,y_unnorm), y_unnorm)
            psnr0 = list(map(add, current_psnr_mu0, psnr0))
            print('Current PSNR', current_psnr_mu0[0])
            psnr0_dens.extend(current_psnr_mu0)
            pd.Series(psnr0_dens).hist()
            plt.xlabel('PSNR')
            plt.ylabel('nr samples')
            plt.savefig(savedir_viz + '/psnr0_density.png', dpi=300, bbox_inches='tight')
            plt.close()

            current_psnr_mu05 = metrics.psnr(inv_scaler(mu05,y_unnorm), y_unnorm)
            psnr05 = list(map(add, current_psnr_mu05, psnr05))

            current_psnr_mu08 = metrics.psnr(inv_scaler(mu08,y_unnorm), y_unnorm)
            psnr08 = list(map(add, current_psnr_mu08, psnr08))

            current_psnr_mu1 = metrics.psnr(mu1, y_unnorm)
            psnr1 = list(map(add, current_psnr_mu1, psnr1))

            # MSE
            # current_mse0 = metrics.MSE(inv_scaler(mu0,y_unnorm), y_unnorm)
            current_mse0 = metrics.MSE(mu0, y)
            mse0 = list(map(add, current_mse0.cpu().numpy(), mse0))
            print('Current MSE', current_psnr_mu0[0])

            current_mse05 = metrics.MSE(inv_scaler(mu05,y_unnorm), y_unnorm)
            mse05 = list(map(add, current_mse05.cpu().numpy(), mse05))

            current_mse08 = metrics.MSE(inv_scaler(mu08,y_unnorm), y_unnorm)
            mse08 = list(map(add, current_mse08.cpu().numpy(), mse08))

            current_mse1 = metrics.MSE(inv_scaler(mu1,y_unnorm),y_unnorm)
            mse1 = list(map(add, current_mse1.cpu().numpy(), mse1))

            # MAE
            # current_mae0 = metrics.MAE(inv_scaler(mu0,y_unnorm),y_unnorm)
            current_mae0 = metrics.MAE(mu0,y)
            mae0 = list(map(add, current_mae0.cpu().numpy(), mae0))
            print('Current MAE', np.mean(mae0_plot))
            mae0_plot.extend(current_mae0.detach().cpu().numpy().tolist())
            pd.Series(mae0_plot).hist()
            plt.xlabel('MAE')
            plt.ylabel('nr samples')
            plt.savefig(savedir_viz + '/mae0_density.png', dpi=300, bbox_inches='tight')
            plt.close()

            current_mae05 = metrics.MAE(mu05,y)
            mae05 = list(map(add, current_mae05.cpu().numpy(), mae05))

            current_mae08 = metrics.MAE(mu08,y)
            mae08 = list(map(add, current_mae08.cpu().numpy(), mae08))

            current_mae1 = metrics.MAE(mu1,y)
            mae1 = list(map(add, current_mae1.cpu().numpy(), mae1))

            # RMSE
            # current_rmse0 = metrics.RMSE(inv_scaler(mu0,y_unnorm),y_unnorm)
            current_rmse0 = metrics.RMSE(mu0,y)
            rmse0 = list(map(add, current_rmse0.cpu().numpy(), mse0))
            print('Current RMSE', current_rmse0[0])

            current_rmse05 = metrics.RMSE(mu05,y)
            rmse05 = list(map(add, current_rmse05.cpu().numpy(), mse05))

            current_rmse08 = metrics.RMSE(mu08,y)
            rmse08 = list(map(add, current_rmse08.cpu().numpy(), mse08))

            current_rmse1 = metrics.RMSE(mu1,y)
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
    avrg_ssim0 = list(map(lambda x: x/len(test_loader), ssim0))
    avrg_ssim05 = list(map(lambda x: x/len(test_loader), ssim05))
    avrg_ssim08 = list(map(lambda x: x/len(test_loader), ssim08))
    avrg_ssim1 = list(map(lambda x: x/len(test_loader), ssim1))

    avrg_psnr0 = list(map(lambda x: x/len(test_loader), psnr0))
    avrg_psnr05 = list(map(lambda x: x/len(test_loader), psnr05))
    avrg_psnr08 = list(map(lambda x: x/len(test_loader), psnr08))
    avrg_psnr1 = list(map(lambda x: x/len(test_loader), psnr1))

    avrg_mse0 = list(map(lambda x: x/len(test_loader), mse0))
    avrg_mse05 = list(map(lambda x: x/len(test_loader), mse05))
    avrg_mse08 = list(map(lambda x: x/len(test_loader), mse08))
    avrg_mse1 = list(map(lambda x: x/len(test_loader), mse1))

    avrg_mae0 = list(map(lambda x: x/len(test_loader), mae0))
    avrg_mae05 = list(map(lambda x: x/len(test_loader), mae05))
    avrg_mae08 = list(map(lambda x: x/len(test_loader), mae08))
    avrg_mae1 = list(map(lambda x: x/len(test_loader), mae1))

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


def metrics_eval(args, model, test_loader, exp_name, modelname, logstep):
    """
    Note: batch size indicates lead time we predict.
    """

    print("Metric evaluation on {}...".format(args.trainset))

    # storing metrics
    ssim = [0] * args.bsz
    psnr = [0] * args.bsz
    mmd = [0] * args.bsz
    emd = [0] * args.bsz
    rmse = [0] * args.bsz

    state = None

    # creat and save metric plots
    savedir = "experiments/{}/plots/test_set_{}/".format(exp_name, args.trainset)
    os.makedirs(savedir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            x = item[0]

            # split time series into lags and prediction window
            x_past, x_for = x[:,:-1,...], x[:,-1,:,:,:].unsqueeze(1)

            x_past = x_past.permute(0,2,1,3,4).contiguous().float().to(args.device)
            x_for = x_for.permute(0,2,1,3,4).contiguous().float().to(args.device)

            z, state, nll = model.forward(x=x_for, x_past=x_past, state=state)

            # track metric over forecasting period
            print("Forecast ... ")
            lead_time = args.bsz-1
            eps = 0.8
            predictions = []
            past = x_past[0,:,:,:,:].unsqueeze(0)

            x, s = model._predict(x_past=past,
                                  state=None,
                                  eps=eps)

            print('COMPUTING ROLLOUT!')
            rollout_len = 0
            stacked_pred, abs_err = create_rollout(model, x, x_for, x_past, s, lead_time)
            x = stacked_pred

            print('ROLLOUT COMPUTED!')

            # # SSIM
            current_ssim = metrics.ssim(x, x_for.squeeze(1))
            ssim = list(map(add, current_ssim, ssim))
            #
            # MMD
            current_mmd = metrics.MMD(x, x_for.squeeze(1))
            mmd = list(map(add, current_mmd.cpu().numpy(), mmd))
            #
            # # PSNR
            current_psnr = metrics.psnr(x, x_for.squeeze(1))
            psnr = list(map(add, current_psnr, psnr))

            # RMSE
            max_value = 315.91873
            min_value = 241.22385
            x_new = x * (max_value - min_value) + min_value
            x_for_new = x_for * (max_value - min_value) + min_value
            current_rmse = metrics.RMSE(x_new.squeeze(1), x_for_new.squeeze(1)) / 10 # divide by ten only for geop data
            rmse = list(map(add, current_rmse.cpu().numpy(), rmse))

            # EMD
            # current_emd = []
            # for i in range(args.bsz):
            #    1

            # current_emd = np.array(current_emd)
            # emd = list(map(add, current_emd, emd))
            # print(ssim[0], psnr[0], mmd[0], emd[0])
            # pdb.set_trace()
            print('3 h', current_rmse[3], current_psnr[3], current_ssim[3])#, emd[0])
            print('20 h', current_rmse[20], current_psnr[20], current_ssim[20])#, emd[0])

            print(batch_idx)
            if batch_idx == 20:
                print(batch_idx)
                break


        # compute average SSIM for each temperature map on predicted day t
        avrg_ssim = list(map(lambda x: x/20, ssim))#len(test_loader), ssim))

        # compute average PSNR for each temperature map on predicted day t
        avrg_psnr = list(map(lambda x: x/20, psnr))#len(test_loader), psnr))

        avrg_mmd = list(map(lambda x: x/20, mmd))#len(test_loader), mmd))

        avrg_emd = list(map(lambda x: x/20, emd))#len(test_loader), emd))

        avrg_rmse = list(map(lambda x: x/20, rmse))#len(test_loader), rmse))

        plt.plot(avrg_ssim, label='ST-Flow Best SSIM', color='deeppink')
        plt.grid(axis='y')
        plt.axvline(x=args.lag_len, color='orangered')
        plt.legend(loc='upper right')
        plt.xlabel('Time-Step')
        plt.ylabel('Average SSIM')
        plt.savefig(savedir + '/avrg_ssim.png', dpi=300)
        plt.close()

        plt.plot(avrg_psnr, label='ST-Flow Best PSNR', color='deeppink')
        plt.grid(axis='y')
        plt.axvline(x=args.lag_len, color='orangered')
        plt.legend(loc='upper right')
        plt.xlabel('Time-Step')
        plt.ylabel('Average PSNR')
        plt.savefig(savedir + '/avrg_psnr.png', dpi=300)
        plt.close()

        plt.plot(avrg_mmd, label='ST-Flow Best MMD', color='deeppink')
        plt.grid(axis='y')
        plt.axvline(x=args.lag_len, color='orangered')
        plt.legend(loc='upper right')
        plt.xlabel('Time-Step')
        plt.ylabel('Average MMD')
        plt.savefig(savedir + '/avrg_mmd.png', dpi=300)
        plt.close()

        plt.plot(avrg_emd, label='ST-Flow Best EMD', color='deeppink')
        plt.grid(axis='y')
        plt.axvline(x=args.lag_len, color='orangered')
        plt.legend(loc='upper right')
        plt.xlabel('Time-Step')
        plt.ylabel('Average EMD')
        plt.savefig(savedir + '/avrg_emd.png', dpi=300)
        plt.close()

        plt.plot(avrg_rmse, label='ST-Flow Best RMSE', color='deeppink')
        plt.grid(axis='y')
        plt.axvline(x=args.lag_len, color='orangered')
        plt.legend(loc='upper right')
        plt.xlabel('Time-Step')
        plt.ylabel('Average RMSE')
        plt.savefig(savedir + '/avrg_rmse.png', dpi=300)
        plt.close()

        # Write metric results to a file in case to recreate plots
        with open(savedir + 'metric_results.txt','w') as f:
            f.write('Avrg SSIM over forecasting period:\n')
            for item in avrg_ssim:
                f.write("%f \n" % item)

            f.write('Avrg PSNR over forecasting period:\n')
            for item in avrg_psnr:
                f.write("%f \n" % item)

            f.write('Avrg MMD over forecasting period:\n')
            for item in avrg_mmd:
                f.write("%f \n" % item)

            f.write('Avrg EMD over forecasting period:\n')
            for item in avrg_emd:
                f.write("%f \n" % item)

            f.write('Avrg RMSE over forecasting period:\n')
            for item in avrg_rmse:
                f.write("%f \n" % item)

        return None

def metrics_eval_all():

    print("Creating unified plot from text files ...")

    # pdb.set_trace()
    path = os.getcwd() + '/experiments/'

    def read_metrics(fname):

        ssim = []
        psnr = []
        rmse = []
        mmd = []
        lines = []

        # read metric results from file
        print('Reading file:', fname)
        with open(path + fname, 'r') as f:
            line = f.readline()

            while line != '':
                print(line, end='')
                line = f.readline()

                if line == 'Avrg MMD over forecasting period:\n':
                    rmse = lines
                    lines = []

                elif line == '':
                    pass

                else:
                    lines.append(float(line))
            mmd = lines

        return rmse, mmd

    # avrg_psnr_l1k8, avrg_ssim_l1k8 = read_metrics('metric_results_flow-1-level-8-k.txt')
    # avrg_psnr_l2k4, avrg_ssim_l2k4 = read_metrics('metric_results_flow-2-level-4-k.txt')
    # avrg_psnr_l3k4, avrg_ssim_l3k4 = read_metrics('metric_results_flow-3-level-4-k.txt')
    # avrg_psnr_3dunet, avrg_ssim_3dunet = read_metrics('metric_results_3dunet.txt')
    avrg_rmse_3dunet, avrg_mmd_3dunet = read_metrics('metric_results_wbench3dunet_30days.txt')
    # avrg_rmse_l3k4, avrg_mmd_l3k4 = read_metrics('metric_results_30daysera5_flow.txt')
    # avrg_rmse_l1k8, avrg_mmd_l1k8 = read_metrics('metric_results_flow_era5_1l8k.txt')
    avrg_rmse_l3k3, avrg_mmd_l3k3 = read_metrics('metric_results_wbench_l3k3.txt')

    # plt.plot(avrg_rmse_l1k8, label='ST-Flow L-1 K-8', color='darkviolet')
    # plt.plot(avrg_rmse_l3k4, label='ST-Flow L-3 K-4', color='deeppink')
    plt.plot(avrg_rmse_l3k3, label='ST-Flow L-3 K-3', color='mediumslateblue')
    plt.plot(avrg_rmse_3dunet, label='3DUnet', color='lightseagreen')
    plt.grid(axis='y')
    plt.axvline(x=2, color='orangered')
    plt.legend(loc='best')
    plt.xlabel('Time-Step')
    plt.ylabel('Average RMSE')
    plt.savefig(path + '/avrg_rmse_all_wbench.png', dpi=300)

    # plt.plot(avrg_ssim_l1k8, label='ST-Flow L1-K-8 Best SSIM', color='darkviolet')
    # plt.plot(avrg_ssim_l3k4, label='ST-Flow L3-K-4 Best SSIM', color='deeppink')
    # plt.plot(avrg_ssim_l2k4, label='ST-Flow L2-K-4 Best SSIM', color='mediumslateblue')
    # plt.plot(avrg_ssim_3dunet, label='3DUnet Best SSIM', color='lightseagreen')
    # plt.grid(axis='y')
    # plt.axvline(x=1, color='orangered')
    # plt.legend(loc='upper right')
    # plt.xlabel('Time-Step')
    # plt.ylabel('Average SSIM')
    # plt.savefig(path + '/avrg_ssim.png', dpi=300)
    # plt.show()

    # plt.plot(avrg_psnr_l1k8, label='ST-Flow L1-K-8 Best PSNR', color='darkviolet')
    # plt.plot(avrg_psnr_l2k4, label='ST-Flow L2-K-4 Best PSNR', color='deeppink')
    # plt.plot(avrg_psnr_l3k4, label='ST-Flow L3-K-4 Best PSNR', color='mediumslateblue')
    # plt.plot(avrg_psnr_3dunet, label='3DUnet Best PNSR', color='lightseagreen')
    # plt.grid(axis='y')
    # plt.axvline(x=1, color='orangered')
    # plt.legend(loc='upper right')
    # plt.xlabel('Time-Step')
    # plt.ylabel('Average PSNR')
    # plt.savefig(path + '/avrg_psnr.png', dpi=300)
    # plt.show()

    return None

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
    modelname = 'model_epoch_5_step_18250'
    modelpath = '/home/christina/Documents/clim-var-ds-cnf/runs/srflow_era5-TCW_2023_09_14_15_37_30/model_checkpoints/{}.tar'.format(modelname)
    # watercontent 2x upsampling
    # modelname = 'model_epoch_7_step_25250'
    # modelpath = '/home/christina/Documents/clim-var-ds-cnf/runs/srflow_era5-TCW_2023_09_18_16_14_06/model_checkpoints/{}.tar'.format(modelname)

    model = srflow.SRFlow((in_channels, args.height, args.width), args.filter_size, args.L, args.K,
                           args.bsz, args.s, args.nb, args.condch, args.nbits, args.noscale, args.noscaletest)

    print(torch.cuda.device_count())
    ckpt = torch.load(modelpath, map_location='cuda:0')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params {}:  '.format(args.device), params)
    model = model.to(args.device)

    plot_std(model, test_loader, "flow-{}-level-{}-k".format(args.L, args.K), modelname, args)

    print("Evaluate on test split ...")
    test(model, test_loader, "flow-{}-level-{}-k".format(args.L, args.K), modelname, -99999, args)
    # metrics_eval(args, model.cuda(), test_loader, "flow-{}-level-{}-k".format(args.L, args.K), modelname, -99999)
    # metrics_eval_all()
