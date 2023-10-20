from datetime import datetime
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.optim as optim
import os
import json
from skimage.transform import resize

import torch
from torch import nn
from torchvision.models.vgg import vgg16
from torch.autograd import Variable

# Utils
from utils import utils
import numpy as np
import random
import pdb
import torchvision
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from optimization.validation_srgan import validate
from typing import Tuple, Callable
import timeit

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

# code adapted from: https://github.com/lizhuoq/SRGAN/blob/main/loss.py

class MinMaxScaler:
    def __call__(self, x, max_value, min_value):
        values_range: Tuple[int, int] = (-1, 1)
        x = (x - min_value) / (max_value - min_value)
        return x * (values_range[1] - values_range[0]) + values_range[0]

def inv_scaler(x):
    min_value = 0
    max_value = 100
    return x * (max_value - min_value) + min_value

class GeneratorLoss(nn.Module):
    def __init__(self,loss_network):
        super(GeneratorLoss, self).__init__()
        # vgg = vgg16(pretrained=True)
        # loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        # we want real_out to be close 1, and fake_out to be close 0
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        return self.tv_loss_weight * 0.5 * (
            torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean() +
            torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        )

def trainer(args, train_loader, valid_loader, model,
            device='cpu', needs_init=True):

    cmap = 'viridis' if args.trainset == 'era5-TCW' else 'inferno'
    config_dict = vars(args)

    # wandb.init(project="arflow", config=config_dict)
    args.experiment_dir = os.path.join('runs',
                                        args.modeltype + '_' + args.trainset  + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"))
    os.makedirs(args.experiment_dir, exist_ok=True)
    config_dict = vars(args)
    with open(args.experiment_dir + '/configs.txt', 'w') as f:
        for key, value in config_dict.items():
            f.write('%s:%s\n' % (key, value))

    # set viz dir
    viz_dir = "{}/snapshots/trainset/".format(args.experiment_dir)
    os.makedirs(viz_dir, exist_ok=True)

    writer = SummaryWriter("{}".format(args.experiment_dir))
    prev_loss_epoch = np.inf
    logging_step = 0
    step = 0

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=2 * 10 ** 5,
    #                                             gamma=0.5)

    metric_dict = {'MSE': [], 'RMSE': [], 'MAE': []}

    generator = model[0].to(device).train()
    discriminator = model[1].to(device).train()
    optimizerG = optim.Adam(generator.parameters(), lr=args.lr, amsgrad=True)
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, amsgrad=True)

    paramsG = sum(x.numel() for x in generator.parameters() if x.requires_grad)
    paramsD = sum(x.numel() for x in discriminator.parameters() if x.requires_grad)
    params = paramsG + paramsD

    print('Nr of Trainable Params on {}:  '.format(device), params)

    # add hyperparameters to tensorboardX logger
    writer.add_hparams({'lr': args.lr, 'bsize':args.bsz, 'Flow Steps':args.K,
                        'Levels':args.L}, {'nll_train': - np.inf})


    if torch.cuda.device_count() > 1 and args.train:
        print("Running on {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        args.parallel = True

    generator_criterion = GeneratorLoss(generator)

    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):

            y = Variable(item[0].to(device))
            x = Variable(item[1].to(device))
            y_unorm = item[2].to(device)
            x_unorm = item[3].to(device)

            generator.zero_grad()
            fake_img=generator(x)

            discriminator.zero_grad()

            fake_out = discriminator(fake_img).mean()
            real_out = discriminator(y).mean()

            d_loss = 1 - real_out + fake_out

            # update discriminator network parameters
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            # update generator network parameters
            g_loss = generator_criterion(fake_out, fake_img, y)
            g_loss.backward()
            optimizerG.step()

            print("[{}] Epoch: {}, Train Step: {:01d}/{}, Bsz = {}, Gen Loss {:.3f}, Disc Loss: {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    epoch, step,
                    args.max_steps,
                    args.bsz,
                    g_loss.mean(),
                    d_loss.mean()))

            if step % args.log_interval == 0:

                with torch.no_grad():

                    generator.eval()
                    discriminator.eval()

                    # Visualize low resolution GT
                    grid_low_res = torchvision.utils.make_grid(x[0:9, :, :, :].cpu(), nrow=3)
                    plt.figure()
                    plt.imshow(grid_low_res.permute(1, 2, 0)[:,:,0], cmap=cmap)
                    plt.axis('off')
                    plt.title("Low-Res GT (train)")
                    # plt.show()
                    plt.savefig(viz_dir + '/low_res_gt{}.png'.format(step), dpi=300, bbox_inches='tight')
                    plt.close()

                    # Visualize High-Res GT
                    grid_high_res_gt = torchvision.utils.make_grid(y[0:9, :, :, :].cpu(), nrow=3)
                    plt.figure()
                    plt.imshow(grid_high_res_gt.permute(
                    1, 2, 0)[:,:,0], cmap=cmap)
                    plt.axis('off')
                    plt.title("High-Res GT")
                    # plt.show()
                    plt.savefig(viz_dir + '/high_res_gt_{}.png'.format(step), dpi=300, bbox_inches='tight')
                    plt.close()

                    # Super-Resolving low-res
                    start = timeit.default_timer()
                    y_hat=generator(x)
                    stop = timeit.default_timer()
                    print("Time Fwd pass:", stop-start)
                    print(y_hat.max(), y_hat.min(), y.max(), y.min())
                    grid_y_hat = torchvision.utils.make_grid(y_hat[0:9, :, :, :].cpu(), nrow=3)
                    plt.figure()
                    plt.imshow(grid_y_hat.permute(1, 2, 0)[:,:,0], cmap=cmap)
                    plt.axis('off')
                    plt.title("Y hat")
                    plt.savefig(viz_dir + '/y_hat{}.png'.format(step), dpi=300,bbox_inches='tight')
                    # plt.show()
                    plt.close()

                    abs_err = torch.abs(y_hat - y)
                    grid_abs_error = torchvision.utils.make_grid(abs_err[0:9,:,:,:].cpu(), nrow=3)
                    plt.figure()
                    plt.imshow(grid_abs_error.permute(1, 2, 0)[:,:,0], cmap=cmap)
                    plt.axis('off')
                    plt.title("Abs Err")
                    plt.savefig(viz_dir + '/abs_err_{}.png'.format(step), dpi=300,bbox_inches='tight')
                    # plt.show()
                    plt.close()

            if step % args.val_interval == 0:
                print('Validating model ... ')

                loss_valid, metric_dict = validate(discriminator, generator,
                                      valid_loader,
                                      metric_dict,
                                      args.experiment_dir,
                                      "{}".format(step),
                                      args)

                writer.add_scalar("loss_valid",
                                  loss_valid.mean().item(),
                                  logging_step)

                # save checkpoint only when nll lower than previous model
                if loss_valid < prev_loss_epoch:
                    PATH = args.experiment_dir + '/model_checkpoints/'
                    os.makedirs(PATH, exist_ok=True)
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_valid.mean()}, PATH+ f"model_epoch_{epoch}_step_{step}.tar")
                    prev_loss_epoch = loss_valid

            logging_step += 1

            if step == args.max_steps:
                break

        if step == args.max_steps:
        #     print("Done Training for {} mini-batch update steps!".format(args.max_steps)
        #     )
        #
        #     if hasattr(model, "module"):
        #         model_without_dataparallel = model.module
        #     else:
        #         model_without_dataparallel = model

            utils.save_model(model,
                             epoch, optimizer, args, time=True)

            print("Saved trained model :)")
            wandb.finish()
            break
