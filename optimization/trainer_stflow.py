from datetime import datetime
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.optim as optim
import os
import json

# Utils
from utils import utils
import numpy as np
import random
import pdb
import torchvision
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from models.architectures.conv_lstm import *
from optimization.validation_stflow import validate

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


def plot_density(density_func, args):

    # set viz dir
    viz_dir = "{}/snapshots/trainset/".format(args.experiment_dir)
    os.makedirs(viz_dir, exist_ok=True)

    tensor = torch.randn(16, 4, 2, 16, 32)

    # Step 1: Create meshgrid
    grid_x, grid_y, grid_z, grid_w, grid_h = torch.meshgrid(
        torch.linspace(-1, 1, tensor.size(0)),  # Define your ranges appropriately
        torch.linspace(-1, 1, tensor.size(1)),  # Define your ranges appropriately
        torch.linspace(-1, 1, tensor.size(2)),  # Define your ranges appropriately
        torch.linspace(-1, 1, tensor.size(3)),   # Define your ranges appropriately
        torch.linspace(-1, 1, tensor.size(4))
    )

    # Concatenate grid points to have shape [num_samples, 4, 2, 16, 32]
    meshgrid_h = grid_x # torch.stack((grid_x, grid_w, , grid_y, grid_z), dim=-1)

    # Step 2: Reshape meshgrid to have shape [16, 4, 2, 16, 32]
    meshgrid_h = meshgrid_h.permute(3, 2, 1, 0, 4).contiguous()

    # Step 1: Create meshgrid
    tensor = torch.randn(16, 4, 1, 16, 32)
    grid_x, grid_y, grid_z, grid_w, grid_h = torch.meshgrid(
        torch.linspace(-1, 1, tensor.size(0)),  # Define your ranges appropriately
        torch.linspace(-1, 1, tensor.size(1)),  # Define your ranges appropriately
        torch.linspace(-1, 1, tensor.size(2)),  # Define your ranges appropriately
        torch.linspace(-1, 1, tensor.size(3)),   # Define your ranges appropriately
        torch.linspace(-1, 1, tensor.size(4))
    )

    meshgrid_z = torch.stack((grid_w, grid_x, grid_y, grid_z), dim=-1)

    # Step 2: Reshape meshgrid to have shape [16, 4, 2, 16, 32]
    meshgrid_z = grid_x #.permute(3, 2, 1, 0, 4).contiguous()

    # Step 3: Evaluate density function
    # Assuming your density function is called 'density_func'
    density_values = density_func(x=meshgrid_z, h=meshgrid_h, reverse=False)

def trainer(args, train_loader, valid_loader, model,
            device='cpu', needs_init=True, ckpt=None):

    config_dict = vars(args)
    # wandb.init(project="arflow", config=config_dict)
    args.experiment_dir = os.path.join('/home/mila/c/christina.winkler/scratch/runs/',
                                        args.modeltype + '_' + args.trainset +'_' + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S") +'_'+ str(args.s)+'x')

    # set viz dir
    viz_dir = "{}/snapshots/trainset/".format(args.experiment_dir)
    os.makedirs(viz_dir, exist_ok=True)

    writer = SummaryWriter("{}".format(args.experiment_dir))
    prev_nll_epoch = np.inf
    logging_step = 0
    step = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=2 * 10 ** 5,
                                                gamma=0.5)
    if args.resume:
        print('Loading optimizer state dict')
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    state=None
    color = 'inferno' if args.trainset == 'temp' else 'viridis'
    model.to(device)

    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params on {}:  '.format(device), params)


    # write training configs to file
    hparams = {'lr': args.lr, 'bsize':args.bsz, 'Flow Steps':args.K, 'Levels':args.L, 's':args.s}

    with open(args.experiment_dir + '/configs.txt','w') as file:
        file.write(json.dumps(hparams))

    if torch.cuda.device_count() > 1 and args.train:
        print("Running on {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        args.parallel = True

    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):

            # retrieve low and high-res
            x, y = item[1].to(device), item[2].to(device)

            # adjust shape of x to match y 
            x = torch.nn.functional.interpolate(x, size=y.shape[-2:], mode='bicubic', align_corners=False)

            x, y = x.unsqueeze(1), y.unsqueeze(1)

            model.train()
            optimizer.zero_grad()

            # We need to init the underlying module in the dataparallel object
            # For ActNorm layers.
            if needs_init and torch.cuda.device_count() > 1:
                bsz_p_gpu = args.bsz // torch.cuda.device_count()
                _, _ = model.module.forward(x_hr=y[:bsz_p_gpu],
                                            xlr=x[:bsz_p_gpu],
                                            logdet=0)

            z, state, nll, logp_z = model.forward(x=y, x_past=x)

            writer.add_scalar("nll_train", nll.mean().item(), step)

            # Compute gradients
            nll.mean().backward()

            # Update model parameters using calculated gradients
            optimizer.step()
            scheduler.step()
            step = step + 1

            print("[{}] Epoch: {}, Train Step: {:01d}/{}, Bsz = {}, NLL {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    epoch, step,
                    args.max_steps,
                    args.bsz,
                    nll.mean()))

            if step % args.log_interval == 0:

                with torch.no_grad():

                    if hasattr(model, "module"):
                        model_without_dataparallel = model.module
                    else:
                        model_without_dataparallel = model

                    model.eval()

                    # testing reconstruction - should be exact same as x_for
                    reconstructions, _, _ = model.forward(z=z.cuda(), x_past=x.cuda(), state=state,
                                                            use_stored=True, reverse=True)

                    squared_recon_error = (reconstructions-y).mean()**2
                    print("Reconstruction Error:", (reconstructions-y).mean())
                    # wandb.log({"Squared Reconstruction Error" : squared_recon_error})

                    grid_reconstructions = torchvision.utils.make_grid(reconstructions[0:9, :, :, :].squeeze(1).cpu(), normalize=True, nrow=3)
                    array_imgs_np = np.array(grid_reconstructions.permute(2,1,0)[:,:,0].contiguous().unsqueeze(2))
                    cmap_recon = np.apply_along_axis(cm.inferno, 2, array_imgs_np)
                    reconstructions = wandb.Image(cmap_recon, caption="Training Reconstruction")
                    # wandb.log({"Reconstructions (train) {}".format(step) : reconstructions})

                    plt.figure()
                    plt.imshow(grid_reconstructions.permute(1, 2, 0)[:,:,0].contiguous(),cmap=color)
                    plt.axis('off')
                    plt.savefig(viz_dir + '/reconstructed_frame_t_{}.png'.format(step), dpi=300)
                    # plt.show()

                    # visualize past frames the prediction is based on (context)
                    grid_past = torchvision.utils.make_grid(x[0:9, -1, :, :].cpu(), normalize=True, nrow=3)
                    array_imgs_past = np.array(grid_past.permute(2,1,0)[:,:,0].contiguous().unsqueeze(2))
                    cmap_past = np.apply_along_axis(cm.inferno, 2, array_imgs_past)
                    past_imgs = wandb.Image(cmap_past, caption="Low-Res")
                    # wandb.log({"Context Frame at t-1 (train) {}".format(step) : past_imgs})

                    plt.figure()
                    plt.imshow(grid_past.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    plt.axis('off')
                    plt.title("Context Frame at t-1 (train)")
                    plt.savefig(viz_dir + '/frame_at_t-1_{}.png'.format(step), dpi=300)

                    # visualize future frame of the correct prediction
                    grid_future = torchvision.utils.make_grid(y[0:9, :, :, :].squeeze(1).cpu(), normalize=True, nrow=3)
                    array_imgs_future = np.array(grid_future.permute(2,1,0)[:,:,0].unsqueeze(2))
                    cmap_future = np.apply_along_axis(cm.inferno, 2, array_imgs_future)
                    future_imgs = wandb.Image(cmap_future, caption="High-Res")
                    # wandb.log({"Frame at t (train) {}".format(step) : future_imgs})

                    plt.figure()
                    plt.imshow(grid_future.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    plt.axis('off')
                    plt.title("Ground Truth at t")
                    plt.savefig(viz_dir + '/frame_at_t_{}.png'.format(step), dpi=300)

                    # visualize log probabilities
                    # plot_density(model.flow.level_modules[-1][-1], args)
                    # logp_z_exp = logp_z.sum(dim=[1]).exp().sum()/ (16*32)
                    # grid_log_pz = torchvision.utils.make_grid(logp_z.sum(dim=[1])[0:9, :, :, :].squeeze(1).cpu(), normalize=True, nrow=3)
                    # plt.figure()
                    # plt.imshow(grid_log_pz.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    # plt.axis('off')
                    # plt.title("Log-probabilities of Gaussianized Input")
                    # plt.show()
                    # plt.savefig(viz_dir + '/log_pz_{}.png'.format(step), dpi=300)

                    # predicting a new sample based on context window
                    print("Predicting ...")
                    predictions, _, _ = model._predict(x.cuda(), state)
                    grid_pred = torchvision.utils.make_grid(predictions[0:9, :, :, :].squeeze(1).cpu(),normalize=True, nrow=3)
                    array_imgs_pred = np.array(grid_pred.permute(2,1,0)[:,:,0].unsqueeze(2))
                    cmap_pred = np.apply_along_axis(cm.inferno, 2, array_imgs_pred)
                    future_pred = wandb.Image(cmap_pred, caption="Prediction")
                    # wandb.log({"Predicted frame at t (train) {}".format(step) : future_pred})

                    # visualize predictions
                    grid_samples = torchvision.utils.make_grid(predictions[0:9, :, :, :].squeeze(1).cpu(),normalize=True, nrow=3)
                    plt.figure()
                    plt.imshow(grid_samples.permute(1, 2, 0)[:,:,0].contiguous(), cmap=color)
                    plt.axis('off')
                    plt.title("Prediction")
                    plt.savefig(viz_dir + '/samples_{}.png'.format(step), dpi=300)


            if step % args.val_interval == 0:
                print('Validating model ... ')
                nll_valid = validate(model_without_dataparallel,
                                     valid_loader,
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
            print("Done Training for {} mini-batch update steps!".format(args.max_steps)
            )

            if hasattr(model, "module"):
                model_without_dataparallel = model.module
            else:
                model_without_dataparallel = model

            utils.save_model(model_without_dataparallel,
                             epoch, optimizer, args, time=True)

            print("Saved trained model :)")
            wandb.finish()
            break
