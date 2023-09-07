import numpy as np
import torch
import random

import PIL
import os
import torchvision
from torchvision import transforms

import sys
sys.path.append("../../")

from os.path import exists, join
import matplotlib.pyplot as plt
import pdb

def validate(model, val_loader, exp_name, logstep, args):

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    state=None
    nll_list=[]
    model.eval()
    # print(len(val_loader))
    with torch.no_grad():
        for batch_idx, item in enumerate(val_loader):

            print(batch_idx)

            x = item[0]

            # split time series into lags and prediction window
            x_past, x_for = x[:,:-1,...], x[:,-1,:,:,:].unsqueeze(1)

            # reshape into correct format for 3D convolutions - but now i dont use them anymore? xD
            x_past = x_past.permute(0,2,1,3,4).contiguous().float()
            x_for = x_for.permute(0,2,1,3,4).contiguous().float()

            z, state, nll = model.forward(x=x_for, x_past=x_past, state=state)

            # Generative loss
            nll_list.append(nll.mean().detach().cpu().numpy())

            if batch_idx == 100:
                break

            # compute metrics
            # TODO: compute MSE
            # TODO: compute negative log predictive density

            # ---------------------- Evaluate Predictions---------------------- #

        # evalutae for different temperatures (just for last batch, perhaps change l8er)
        mu0, _ = model._predict(x_past, state, eps=0)
        mu05, _ = model._predict(x_past, state, eps=0.5)
        mu08, _ = model._predict(x_past, state, eps=0.8)
        mu1, _ = model._predict(x_past, state, eps=1)

        savedir = "{}/snapshots/validationset_{}/".format(exp_name, args.trainset)

        os.makedirs(savedir, exist_ok=True)

        grid_ground_truth = torchvision.utils.make_grid(x_for[0:9, :, :, :].squeeze(1).cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_ground_truth.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
        plt.axis('off')
        plt.title("Frame at t (valid)")
        plt.savefig(savedir + "x_t_step_{}_valid.png".format(logstep), dpi=300)

        # visualize past frames the prediction is based on (context)
        grid_past = torchvision.utils.make_grid(x_past[0:9, -1, :, :].cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_past.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
        plt.axis('off')
        plt.title("Frame at t (valid)")
        plt.savefig(savedir + "_x_t_step_{}_valid.png".format(logstep), dpi=300)

        grid_mu0 = torchvision.utils.make_grid(mu0[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_mu0.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=0")
        plt.savefig(savedir + "mu_0_logstep_{}_valid.png".format(logstep), dpi=300)

        grid_mu05 = torchvision.utils.make_grid(mu05[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_mu0.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=0.5")
        plt.savefig(savedir + "mu_0.5_logstep_{}_valid.png".format(logstep), dpi=300)

        grid_mu08 = torchvision.utils.make_grid(mu08[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_mu08.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=0.8")
        plt.savefig(savedir + "mu_0.8_logstep_{}_valid.png".format(logstep), dpi=300)

        grid_mu1 = torchvision.utils.make_grid(mu1[0:9,:,:,:].squeeze(1).cpu(), nrow=3)
        plt.figure()
        plt.imshow(grid_mu1.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
        plt.axis('off')
        plt.title("Prediction at t (valid), mu=1.0")
        plt.savefig(savedir + "mu_1_logstep_{}_valid.png".format(logstep), dpi=300)

    print("Average Validation Neg. Log Probability Mass:", np.mean(nll_list))
    return np.mean(nll_list)

def mse(arg):
    """
    Implements Mean Squared Error.
    Args:
        prediction
        ground_truth
    """
    pass

def nlpd(arg):
    """
    Implements negative log predictive density.
    """
    pass

def metrics_eval(model, test_loader, logging_step, writer, args):

    print("Metric evaluation on {}...".format(args.testset))

    # storing metrics
    # ssim_yhat = []
    ssim_mu0 = []
    ssim_mu05 = []
    ssim_mu08 = []
    ssim_mu1 = []
    # psnr_yhat = []
    psnr_0 = []
    psnr_05 = []
    psnr_08 = []
    psnr_1 = []

    model.eval()
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):

            y = item[0]
            x = item[1]
            orig_shape = item[2]
            w, h = orig_shape

            # Push tensors to GPU
            y = y.to("cuda")
            x = x.to("cuda")

            if args.modeltype == "flow":
                mu0 = model._sample(x=x, eps=0)
                mu05 = model._sample(x=x, eps=0.5)
                mu08 = model._sample(x=x, eps=0.8)
                mu1 = model._sample(x=x, eps=1)

                ssim_mu0.append(metrics.ssim(y, mu0, orig_shape))
                ssim_mu05.append(metrics.ssim(y, mu05, orig_shape))
                ssim_mu08.append(metrics.ssim(y, mu08, orig_shape))
                ssim_mu1.append(metrics.ssim(y, mu1, orig_shape))

                psnr_0.append(metrics.psnr(y, mu0, orig_shape))
                psnr_05.append(metrics.psnr(y, mu05, orig_shape))
                psnr_08.append(metrics.psnr(y, mu08, orig_shape))
                psnr_1.append(metrics.psnr(y, mu1, orig_shape))

            elif args.modeltype == "dlogistic":
                # sample from model
                sample, means = model._sample(x=x)
                ssim_mu0.append(metrics.ssim(y, means, orig_shape))
                psnr_0.append(metrics.psnr(y, means, orig_shape))

                # ---------------------- Visualize Samples-------------
                if args.visual:
                    # only for testing, delete snippet later
                    torchvision.utils.save_image(
                        x[:, :, :h, :w], "x.png", nrow=1, padding=2, normalize=False
                    )
                    torchvision.utils.save_image(
                        y[:, :, :h, :w], "y.png", nrow=1, padding=2, normalize=False
                    )
                    torchvision.utils.save_image(
                        means[:, :, :h, :w],
                        "dlog_mu.png",
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )
                    torchvision.utils.save_image(
                        sample[:, :, :h, :w],
                        "dlog_sample.png",
                        nrow=1,
                        padding=2,
                        normalize=False,
                    )

        writer.add_scalar("ssim_std0", np.mean(ssim_mu0), logging_step)
        writer.add_scalar("psnr0", np.mean(psnr_0), logging_step)

        if args.modeltype == "flow":
            writer.add_scalar("ssim_std05", np.mean(ssim_mu05), logging_step)
            writer.add_scalar("ssim_std08", np.mean(ssim_mu08), logging_step)
            writer.add_scalar("ssim_std1", np.mean(ssim_mu1), logging_step)
            writer.add_scalar("psnr05", np.mean(psnr_05), logging_step)
            writer.add_scalar("psnr08", np.mean(psnr_08), logging_step)
            writer.add_scalar("psnr1", np.mean(psnr_1), logging_step)

        print("PSNR (GT, mean):", np.mean(psnr_0))
        print("SSIM (GT, mean):", np.mean(ssim_mu0))

        return writer

# if __name__ == "__main__":

    # validate(model, val_loader, exp_name, logstep, args)
    # evaluate_metrics(model, dloader)
