from datetime import datetime
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.optim as optim
import os
from skimage.transform import resize

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

def trainer(args, train_loader, valid_loader, model,
            device='cpu', needs_init=True):

    config_dict = vars(args)
    # wandb.init(project="arflow", config=config_dict)
    args.experiment_dir = os.path.join('runs',
                                        args.modeltype + '_' + args.trainset  + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"))
    # set viz dir
    viz_dir = "{}/snapshots/trainset/".format(args.experiment_dir)
    os.makedirs(viz_dir, exist_ok=True)

    writer = SummaryWriter("{}".format(args.experiment_dir))
    prev_nll_epoch = np.inf
    logging_step = 0
    step = 0
    bpd_valid = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=2 * 10 ** 5,
                                                gamma=0.5)
    state=None

    model.to(device)

    params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('Nr of Trainable Params on {}:  '.format(device), params)

    # add hyperparameters to tensorboardX logger
    writer.add_hparams({'lr': args.lr, 'bsize':args.bsz, 'Flow Steps':args.K,
                        'Levels':args.L}, {'nll_train': - np.inf})


    # pdb.set_trace()

    if torch.cuda.device_count() > 1 and args.train:
        print("Running on {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        args.parallel = True

    for epoch in range(args.epochs):
        for batch_idx, item in enumerate(train_loader):

            y = item[0].to(device).unsqueeze(1)
            x = item[1].to(device).unsqueeze(1)

            model.train()
            optimizer.zero_grad()

            # # We need to init the underlying module in the dataparallel object
            # For ActNorm layers.
            if needs_init and torch.cuda.device_count() > 1:
                bsz_p_gpu = args.bsz // torch.cuda.device_count()
                _, _ = model.module.forward(x_hr=y[:bsz_p_gpu],
                                            xlr=x[:bsz_p_gpu],
                                            logdet=0)

            z, state, nll = model.forward(x=y, x_past=x, state=state)
            writer.add_scalar("nll_train", nll.mean().item(), step)
            # wandb.log({"nll_train": nll.mean().item()}, step)

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
                    # pdb.set_trace()
                    reconstructions, _ = model.forward(z=z.cuda(), x_past=x_past.cuda(), state=state,
                                      use_stored=True, reverse=True)

                    squared_recon_error = (reconstructions-x_for).mean()**2
                    print("Reconstruction Error:", (reconstructions-x_for).mean())
                    # wandb.log({"Squared Reconstruction Error" : squared_recon_error})

                    grid_reconstructions = torchvision.utils.make_grid(reconstructions[0:9, :, :, :].squeeze(1).cpu(), nrow=3)
                    array_imgs_np = np.array(grid_reconstructions.permute(2,1,0)[:,:,0].contiguous().unsqueeze(2))
                    cmap_recon = np.apply_along_axis(cm.inferno, 2, array_imgs_np)
                    reconstructions = wandb.Image(cmap_recon, caption="Training Reconstruction")
                    # wandb.log({"Reconstructions (train) {}".format(step) : reconstructions})

                    plt.figure()
                    plt.imshow(grid_reconstructions.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
                    plt.axis('off')
                    plt.savefig(viz_dir + '/reconstructed_frame_t_{}.png'.format(step), dpi=300)
                    # plt.show()

                    # visualize past frames the prediction is based on (context)
                    grid_past = torchvision.utils.make_grid(x_past[0:9, -1, :, :].cpu(), nrow=3)
                    array_imgs_past = np.array(grid_past.permute(2,1,0)[:,:,0].contiguous().unsqueeze(2))
                    cmap_past = np.apply_along_axis(cm.inferno, 2, array_imgs_past)
                    past_imgs = wandb.Image(cmap_past, caption="Frame at t-1")
                    # wandb.log({"Context Frame at t-1 (train) {}".format(step) : past_imgs})

                    plt.figure()
                    plt.imshow(grid_past.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
                    plt.axis('off')
                    plt.title("Context Frame at t-1 (train)")
                    plt.savefig(viz_dir + '/frame_at_t-1_{}.png'.format(step), dpi=300)
                    #
                    # # visualize future frame of the correct prediction
                    grid_future = torchvision.utils.make_grid(x_for[0:9, :, :, :].squeeze(1).cpu(), nrow=3)
                    array_imgs_future = np.array(grid_future.permute(2,1,0)[:,:,0].unsqueeze(2))
                    cmap_future = np.apply_along_axis(cm.inferno, 2, array_imgs_future)
                    future_imgs = wandb.Image(cmap_future, caption="Frame at t")
                    # wandb.log({"Frame at t (train) {}".format(step) : future_imgs})

                    plt.figure()
                    plt.imshow(grid_future.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
                    plt.axis('off')
                    plt.title("Ground Truth at t")
                    plt.savefig(viz_dir + '/frame_at_t_{}.png'.format(step), dpi=300)

                     # predicting a new sample based on context window
                    print("Predicting ...")
                    predictions, _ = model._predict(x_past.cuda(), state) # TODO: sample longer trajectories
                    grid_pred = torchvision.utils.make_grid(predictions[0:9, :, :, :].squeeze(1).cpu(), nrow=3)
                    array_imgs_pred = np.array(grid_pred.permute(2,1,0)[:,:,0].unsqueeze(2))
                    cmap_pred = np.apply_along_axis(cm.inferno, 2, array_imgs_pred)
                    future_pred = wandb.Image(cmap_pred, caption="Frame at t")
                    # wandb.log({"Predicted frame at t (train) {}".format(step) : future_pred})

                    # visualize predictions
                    grid_samples = torchvision.utils.make_grid(predictions[0:9, :, :, :].squeeze(1).cpu(), nrow=3)
                    plt.figure()
                    plt.imshow(grid_samples.permute(1, 2, 0)[:,:,0].contiguous(), cmap='inferno')
                    plt.axis('off')
                    plt.title("Prediction at t")
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
