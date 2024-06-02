# Import libraries
from datetime import datetime
import os
import json
import random
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
from matplotlib import cm

from skimage.transform import resize

import torchvision
from tensorboardX import SummaryWriter

import wandb
os.environ["WANDB_SILENT"] = "true"

# Add project directory to sys.path for relative imports
import sys
sys.path.append("../../")

# Custom Utils
from utils import utils
from optimization.validation_srflow import validate
from typing import Tuple, Callable

class MinMaxScaler:
    def __call__(self, x, max_value, min_value):
        values_range: Tuple[int, int] = (-1, 1)
        x = (x - min_value) / (max_value - min_value)
        return x * (values_range[1] - values_range[0]) + values_range[0]

def inv_scaler(x):
    min_value = 0
    max_value = 100
    return x * (max_value - min_value) + min_value

def inverse_normalize_data_in_window(x_norm, w_min, w_max):
    '''Inverts the linear operation included in normalize_data_in_window. Cannot handle data clipping. :paramx_norm: Normalized image. :paramw_min: Lower bound of window. :paramw_max: Upper bound of window. :return: Tensor. '''
    return (x_norm * (w_max - w_min)) + w_min

def trainer(args, train_loader, valid_loader, model, opt, device='cpu'):

    cmap = 'viridis' if args.trainset == 'era5-TCW' else 'inferno'
    config_dict = vars(args)

    args.experiment_dir = os.path.join('runs',
                                        args.modeltype + '_' + args.trainset + '_' + args.constraint + '_' + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S") +'_'+ str(args.s)+'x')

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

    # Train
    current_step = model.begin_step
    current_epoch = model.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    model.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    model.to(device)
    scaler = MinMaxScaler()
    l1 = nn.L1Loss()
    metric_dict = {'MSE': [], 'RMSE': [], 'MAE': []}
    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params on {}:  '.format(device), params)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    # add hyperparameters to tensorboardX logger
    #writer.add_hparams({'lr': args.lr, 'bsize':args.bsz, 'nll_train': - np.inf})

    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):

            y = item[0].to(device)
            x = item[1].to(device)
            # y_unorm = item[2].to(device)
            # x_unorm = item[3].to(device)

            model.train()
            optimizer.zero_grad()

            # forward loss
            # interpolate x to get SR
            sr = F.interpolate(x, scale_factor=args.s, mode='bilinear')

            data = {'HR':y, 'SR':sr, 'LR':x}
            model.feed_data(data)
            model.optimize_parameters()

            # if step % 20==0:
            #     plt.figure(figsize=(6, 4))
            #     plt.imshow(data['SR'][0,...].permute(1, 2, 0).cpu().numpy())
            #     plt.title('Shape of SR Tensor')
            #     plt.show()

            if current_step % 20 == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_epoch, current_step)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                print(message)

            step = step + 1

            model.test(continous=False)
            visuals = model.get_current_visuals()
            if step % 4==0:
                plt.figure(figsize=(6, 4))
                plt.imshow(visuals['SR'].permute(1, 2, 0).cpu().numpy())
                plt.title('Super-Resolved Image')
                plt.savefig(viz_dir + '/y_hat{}.png'.format(step), dpi=300,bbox_inches='tight')
                plt.axis('off')
                # plt.show()

            if step % args.log_interval == 0:

                with torch.no_grad():

                    # if hasattr(model, "module"):
                    #     model_without_dataparallel = model.module
                    # else:
                    #     model_without_dataparallel = model

                    model.eval()

                    # Visualize low resolution GT
                    grid_low_res = torchvision.utils.make_grid(x[0:9, :, :, :].cpu(), normalize=True, nrow=3)
                    plt.figure()
                    plt.imshow(grid_low_res.permute(1, 2, 0)[:,:,0], cmap=cmap)
                    plt.axis('off')
                    plt.title("Low-Res GT (train)")
                    # plt.show()
                    plt.savefig(viz_dir + '/low_res_gt{}.png'.format(step), dpi=300, bbox_inches='tight')
                    plt.close()

                    # Visualize High-Res GT
                    grid_high_res_gt = torchvision.utils.make_grid(y[0:9, :, :, :].cpu(), normalize=True, nrow=3)
                    plt.figure()
                    plt.imshow(grid_high_res_gt.permute(1, 2, 0)[:,:,0], cmap=cmap)
                    plt.axis('off')
                    plt.title("High-Res GT")
                    # plt.show()
                    plt.savefig(viz_dir + '/high_res_gt_{}.png'.format(step), dpi=300, bbox_inches='tight')
                    plt.close()

                     # Super-Resolving low-res
                    y_hat, logdet, logpz = model(xlr=x, reverse=True, eps=0.8)
                    # print(y_hat.max(), y_hat.min(), y.max(), y.min())
                    grid_y_hat = torchvision.utils.make_grid(y_hat[0:9, :, :, :].cpu(), normalize=False, nrow=3)
                    plt.figure()
                    plt.imshow(grid_y_hat.permute(1, 2, 0)[:,:,0], cmap=cmap)
                    plt.axis('off')
                    plt.title("Y hat")
                    plt.savefig(viz_dir + '/y_hat_mu08_{}.png'.format(step), dpi=300,bbox_inches='tight')
                    # plt.show()
                    plt.close()

                    abs_err = torch.abs(y_hat - y)
                    grid_abs_error = torchvision.utils.make_grid(abs_err[0:9,:,:,:].cpu(), normalize=True, nrow=3)
                    plt.figure()
                    plt.imshow(grid_abs_error.permute(1, 2, 0)[:,:,0], cmap=cmap)
                    plt.axis('off')
                    plt.title("Abs Err")
                    plt.savefig(viz_dir + '/abs_err_{}.png'.format(step), dpi=300,bbox_inches='tight')
                    # plt.show()
                    plt.close()


            if step % args.val_interval == 0:
                print('Validating model ... ')
                metric_dict, nll_valid = validate(model,
                                     valid_loader,
                                     metric_dict,
                                     args.experiment_dir,
                                     "{}".format(step),
                                     args)

                writer.add_scalar("nll_valid",
                                  nll_valid.mean().item(),
                                  logging_step)

                # save checkpoint only when nll lower than previous model
                if nll_valid < prev_nll_epoch:
                    PATH = args.experiment_dir + '/model_checkpoints/'
                    os.makedirs(PATH, exist_ok=True)
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': nll_valid.mean()}, PATH+ f"model_epoch_{epoch}_step_{step}.tar")
                    prev_nll_epoch = nll_valid

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/dtsr_sinogram.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-scale',type=int,default=4)


    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    scale = args.scale
    print(f"Training for a scale of {scale}")


    #Train and Validation dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            # train_set = Data.create_dataset(dataset_opt, phase)          #Add our data here
            train_set = SinogramData(data_dir = './Sinogram/Data/Train',patch_size = 128, scale_factor = scale,image_type = '.tif',image_channels=1)
            train_loader = Data.create_dataloader(                              #And here
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = SinogramData(data_dir = './Sinogram/Data/Valid',patch_size = 256, scale_factor = scale,image_type = '.tif',image_channels=1,training = False)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)

    # Test Data
    test_data = SinogramData(data_dir = './Sinogram/Data/Test',patch_size = 256, scale_factor = scale,image_type = '.tif',image_channels=1,training = False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)


                validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    srtb ,lrtb, hrtb, fktb = [],[],[],[]
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()

                        sr_img = inverse_normalize_data_in_window(visuals['SR'],-1,13)
                        hr_img = inverse_normalize_data_in_window(visuals['HR'],-1,13)
                        lr_img = inverse_normalize_data_in_window(visuals['LR'],-1,13)
                        fake_img = inverse_normalize_data_in_window(visuals['INF'],-1,13)
                        lr_image = F.interpolate(lr_img, scale_factor=4)

                        srtb.append(torch.unsqueeze(sr_img,dim=0))
                        lrtb.append(lr_image)
                        hrtb.append(hr_img)
                        fktb.append(fake_img)
                        sr_img = sr_img.numpy()
                        hr_img = hr_img.numpy()
                        lr_img = lr_img.numpy()
                        fake_img = fake_img.numpy()


                        Metrics.save_to_tiff_stack(
                            sr_img, Path('{}/{}_{}_sr.tif'.format(result_path, current_step, idx)))
                        Metrics.save_to_tiff_stack(
                            hr_img, Path('{}/{}_{}_hr.tif'.format(result_path, current_step, idx)))
                        Metrics.save_to_tiff_stack(
                            lr_img, Path('{}/{}_{}_lr.tif'.format(result_path, current_step, idx)))
                        Metrics.save_to_tiff_stack(
                            fake_img, Path('{}/{}_{}_inf.tif'.format(result_path, current_step, idx)))

                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)

                    lrtb = torch.cat(lrtb,dim=0)
                    hrtb = torch.cat(hrtb,dim=0)
                    fktb = torch.cat(fktb,dim=0)
                    srtb = torch.cat(srtb,dim=0)

                    avg_psnr = avg_psnr / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    val_step += 1


                #

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)



        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        invTrans = transform_lib.Compose([ transform_lib.Normalize(mean = [ 0.0],
                                                     std = [ 1.0]),
                                transform_lib.Normalize(mean = [ 0.0],
                                                     std = [ 1.0]),
                               ])
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  test_data in enumerate(test_loader):
            idx += 1
            diffusion.feed_data(test_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()
            hr_img = invTrans(visuals['HR'])
            # hr_img = visuals['HR'].numpy()
            hr_img = hr_img.numpy()
            lr_img = visuals['LR'].numpy()
            fake_img = visuals['INF'].numpy()

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                sr_img = invTrans(visuals['SR'])
                sr_img = sr_img.numpy()
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_to_tiff_stack(
                        sr_img[iter], Path('{}/{}_{}_sr_{}.tif'.format(result_path, current_step, idx,iter))
                    )
            else:
                sr_img = invTrans(visuals['SR'])
                sr_img = sr_img.numpy()
                Metrics.save_to_tiff_stack(
                    sr_img, Path('{}/{}_{}_sr_process.tif'.format(result_path, current_step, idx)))

            Metrics.save_to_tiff_stack(
                sr_img[-1], Path('{}/{}_{}_sr.tif'.format(result_path, current_step, idx)))
            Metrics.save_to_tiff_stack(
                hr_img, Path('{}/{}_{}_hr.tif'.format(result_path, current_step, idx)))
            Metrics.save_to_tiff_stack(
                lr_img, Path('{}/{}_{}_lr.tif'.format(result_path, current_step, idx)))
            Metrics.save_to_tiff_stack(
                fake_img, Path('{}/{}_{}_inf.tif'.format(result_path, current_step, idx)))
            eval_psnr = psnr(visuals['SR'][-1].unsqueeze(0), visuals['HR'])
            eval_ssim = ssim(visuals['SR'][-1].unsqueeze(0), visuals['HR'])

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim


        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))
