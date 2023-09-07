import torch
import torch.nn as nn
import numpy as np
import random

from models.transformations import modules
from models.architectures.conv_lstm import ConvLSTMCell
from models.transformations.dequantization import *

import pdb
import sys
sys.path.append("../../")

# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)
#
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class FlowStep(nn.Module):

    def __init__(self, level, s, channel_dim, input_shape, filter_size,
                 cond_channels, lag_len,noscale, noscaletest, testmode):

        super().__init__()

        self.channel_dim = channel_dim

        # 1. Activation Normalization
        self.actnorm = modules.ActNorm(channel_dim, testmode=testmode)

        # 2. Invertible 1x1 Convolution
        self.invconv = modules.Invert1x1Conv(channel_dim)

        # 3. Conditional Coupling layer
        self.conditionalCoupling = modules.ConditionalCoupling(level,
                                                               s, channel_dim,
                                                               input_shape,
                                                               cond_channels,
                                                               lag_len,
                                                               filter_size,
                                                               noscale,
                                                               noscaletest)

    def forward(self, z, h=None, x_lr=None, logdet=0, reverse=False):

        if not reverse:

            # 1. Activation normalization layer
            # print("shape of z", z.shape)
            z, logdet = self.actnorm(z.squeeze(2), logdet=logdet, reverse=False)

            # 2. Permutation with invertible 1x1 Convolutional layer
            z, logdet = self.invconv.forward_(z, logdet=logdet, reverse=False)

            # 3. Conditional Coupling Operation
            z = z.unsqueeze(2)
            z, logdet = self.conditionalCoupling(z=z, h=h,
                                                 logdet=logdet, reverse=False)

            return z, logdet

        else:
            # 1. Conditional Coupling
            z, logdet = self.conditionalCoupling(z, h=h,
                                                 logdet=logdet, reverse=True)
            # 2. Invertible 1x1 convolution
            z = z.squeeze(2)
            z, logdet = self.invconv.forward_(z, logdet=logdet, reverse=True)

            # 3. Actnorm
            z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
            z = z.unsqueeze(2)

            return z, logdet

class NormFlowNet(nn.Module):
    def __init__(self, input_shape, filter_size, bsz, s, L, K, lag_len,
                 nb, cond_channels, noscale, noscaletest, device, testmode):

        super().__init__()
        self.L = L
        self.K = K
        self.bsz = bsz
        C, H, W = input_shape
        self.lag_len = lag_len
        self.input_shape = input_shape
        self.testmode = testmode
        self.output_shapes = []
        self.layers = nn.ModuleList()
        self.conv_lstm = ConvLSTMCell(in_channels=2, hidden_channels=32,
                                      out_channels=4*1, num_layers=3).to('cuda')
        self.device = device
        self.init = True
        self.reduce_channels = nn.Conv3d(4, 2, kernel_size=3, padding=1)

        # C for reshaping, C_modules for keeping track of channel dim inside invert transforms

        # Build Normalizing Flow
        self.level_modules = torch.nn.ModuleList()

        for i in range(self.L):
            self.level_modules.append(nn.ModuleList())

        for i in range(self.L):

            # 1. Squeeze Layer
            self.level_modules[i].append(modules.SqueezeTS(factor=2))

            C, H, W = C * 4, H // 2, W // 2
            self.output_shapes.append((bsz, C, lag_len, H, W))
            # C_modules = C if i <= 0 else C // 2

            if i <= 0:
                C_modules = C
            elif i == 1:
                C_modules = C // 2
            elif i == 2:
                C_modules = C // 4

            # 2. Flow Steps
            for k in range(K):
                self.level_modules[i].append(FlowStep(i, s, C_modules, input_shape,
                                             filter_size, cond_channels, lag_len,
                                             noscale, noscaletest, testmode)).to('cuda')

            if i < L - 1:
                # 3.Split Prior for intermediate latent variables
                self.level_modules[i].append(modules.GaussianPrior(C_modules, s, i, cond_channels, lag_len, (bsz, C_modules, lag_len, H, W)))
                # C_prior = C // 2
                # self.output_shapes.append((bsz, C, lag_len, H, W))

        self.level_modules[-1].append(modules.GaussianPrior(C_modules, s, i, cond_channels, lag_len,
                                     (bsz, C_modules, lag_len, H, W), final=True))

        self.extra_squeezer = modules.SqueezeTS(factor=2)
        C, H, W = self.input_shape
        self.output_shapes.append((bsz, C, lag_len, H, W))

    def forward(self, z=None, x_past=None, state=None, logdet=0,
                logpz=0, eps=None, reverse=False, use_stored=False):

        if self.init or state==None:
            state = (torch.zeros_like(x_past), torch.zeros_like(x_past))
            (h, c) = self.conv_lstm(x_past, state)
            self.init = False

        else:
            (h, c) = self.conv_lstm(x_past, state)

        # Encode
        if not reverse:
            for i in range(self.L):

                for layer in self.level_modules[i]:

                    if isinstance(layer, modules.SqueezeTS):
                        # print("Squeezy")

                        bsz, channel, laglen, height, width = self.output_shapes[i]
                        if i <= 0:
                            C = channel
                        elif i == 1:
                            C = channel // 2
                        elif i==2:
                            C = self.output_shapes[i-1][1]

                        z_shaped = torch.zeros(bsz, C, 1, height, width).to(self.device)
                        h_shaped = torch.zeros(self.output_shapes[i]).to(self.device)

                        # squeeze each time-frame seperately
                        if i > 0:
                            for s in range(self.lag_len):
                                h_shaped[:,:,s,:,:] = layer(h[:,:,s,:,:], reverse=False)
                                z_shaped[:,:,0,:,:] = layer(z[:,:,0,:,:], reverse=False)

                            z = z_shaped
                            h = h_shaped

                        else:
                            h_shaped = h.repeat(1,4,1,1,1)
                            for s in range(self.lag_len):
                                # h_shaped[:,:,s,:,:] = layer(h[:,:,s,:,:], reverse=False)
                                z_shaped[:,:,0,:,:] = layer(z[:,:,0,:,:], reverse=False)

                            z = z_shaped
                            h = h_shaped

                    elif isinstance(layer, FlowStep):
                        # print("FlowStep at Level:", i)
                        z, logdet = layer(z, h=h,
                                          logdet=logdet, reverse=False)

                    elif isinstance(layer, modules.GaussianPrior):
                        # print("GaussianPrior")
                        z, logdet, logpz = layer(z, logdet=logdet, logpz=logpz,
                                                 h=h_shaped,
                                                 eps=eps, reverse=False)

        # Decode
        else:

            # bring context state to correct shape at last level
            for i in range(self.L):
                # pdb.set_trace()
                bsz, c, t, height, w = self.output_shapes[i]
                # bsz = 1 if self.testmode else bsz
                h_temp = torch.zeros((bsz,c,t,height,w)).to(self.device).cuda()

                if i > 0:
                    for l in range(self.lag_len):
                        h_temp[:,:,l,:,:] = self.extra_squeezer(h[:,:,l,:,:])
                    h = h_temp.clone()
                else:
                    h = h.repeat(1,4,1,1,1)

            for i in reversed(range(self.L)):
                for layer in reversed(self.level_modules[i]):

                    if isinstance(layer, modules.GaussianPrior):

                        z, logdet, logpz = layer(z, h=h,
                                                logdet=logdet, logpz=logpz, eps=eps,
                                                reverse=True, use_stored=use_stored)

                    elif isinstance(layer, FlowStep):
                        z, logdet = layer(z=z, h=h,
                                          logdet=logdet, reverse=True)

                    elif isinstance(layer, modules.SqueezeTS):
                        z = z.squeeze(2)
                        z = layer(z, reverse=True)
                        z = z.unsqueeze(2)
                        bsz, c, t, height, w = self.output_shapes[i-1]
                        # bsz = 1 if self.testmode else bsz
                        h_temp = torch.zeros((bsz,c,t,height,w)).to(self.device)

                        for l in range(self.lag_len):
                            h_temp[:,:,l,:,:] = self.extra_squeezer(h[:,:,l,:,:], reverse=True)
                        h = h_temp.clone()

        return z, state, logdet, logpz

class FlowModel(nn.Module):
    def __init__(self, input_shape, filter_size, L, K, bsz, lag_len,
                 s, nb, device='cpu', cond_channels=128,
                 n_bits_x=8, noscale=False, noscaletest=False, testmode=False):

        super().__init__()

        self.flow = NormFlowNet(input_shape=input_shape, filter_size=filter_size,
                                s=s, bsz=bsz,K=K, lag_len=lag_len, L=L, nb=nb,
                                cond_channels=cond_channels, device=device,
                                noscale=noscale, noscaletest=noscaletest,
                                testmode=testmode)

        # varflows = [modules.CouplingLayer(network=modules.GatedConvNet(c_in=2*input_shape[0], c_out=2*input_shape[0], c_hidden=16),
        #             mask=modules.create_checkerboard_mask(h=input_shape[1], w=input_shape[2], invert=(i%2==1)),
        #             c_in=input_shape[0]) for i in range(4)]

        # self._variational_dequantizer = VariationalDequantization(varflows)
        # self.nbins = 2 ** n_bits_x

    def forward(self, x=None, x_past=None, z=None, logdet=0, state=None,
                eps=None, reverse=False, use_stored=False):

        if not reverse:
            return self.normalizing_flow(x=x, x_past=x_past, state=state)

        else:
            return self.inverse_flow(z=z, x_past=x_past, state=state,
                                     eps=eps, use_stored=use_stored)

    def normalizing_flow(self, x, x_past, state=None):

        logdet = torch.zeros_like(x[:, 0, 0, 0, 0])
        z=x

        # Push z through flow
        z, state, logdet, logp_z = self.flow.forward(z=z, x_past=x_past,
                                                     logdet=logdet,
                                                     state=state)
        # Loss: Z'ks under Gaussian + sum_logdet
        # pdb.set_trace()
        # D =  float(np.log(2) * np.prod(x.size()[1:]))
        nll = -(logdet + logp_z) #/ D
        return z, state, nll

    def inverse_flow(self, x_past, state, z=None, eps=1.0, use_stored=False):
        x, state, logdet, logpz = self.flow.forward(x_past=x_past, state=state,
                                                    z=z, eps=eps, reverse=True,
                                                    use_stored=use_stored)

        # pass reverse through variational dequantizer
        # x, _ = self._variational_dequantizer(x, ldj=logdet.unsqueeze(0), reverse=True)
        return x, state

    def sigmoid(self, z, ldj, alpha=1e-5, reverse=False):
        """
        Implements the sigmoid function as bijector.
        Args:
            z - tensor to be transformed.
            ldj - logdeterminant to keep track of volume changes.
            alpha - small constant that is used to scale the original input.
        """

        if not reverse: # sigmoid
            ldj += (-z-2*F.softplus(-z)).sum(dim=[0,1])
            z = torch.sigmoid(z)

        else: # reverse mode (inverse sigmoid)
            z = z * (1 - alpha) + 0.5 * alpha  # Scale to prevent boundaries 0 and 1
            # pdb.set_trace()
            ldj += np.log(1 - alpha) * np.prod(z.shape[0:])
            ldj += (-torch.log(z) - torch.log(1-z)).sum(dim=[0,1])
            z = torch.log(z) - torch.log(1-z)

        return z, ldj

    def softplus(self, z, ldj, alpha=1e-5, reverse=False):
        """
        Implements the softplus function as bijector.
        alpha - small constant that is used to scale the original input.
        """

        if not reverse: # softplus
            raise NotImplementedError
            #ldj += TODO ??? (-z-2*F.softplus(-z)).sum(dim=[0,1])
            # z = torch.softplus(z)

        else: # reverse mode (inverse softplus)
            raise NotImplementedError
            # z = z * (1 - alpha) + 0.5 * alpha  # Scale to prevent boundaries 0 and 1
            # ldj += np.log(1 - alpha) * np.prod(z.shape[1:])
            # ldj += (-torch.log(z) - torch.log(1-z)).sum(dim=[0,1])
            # z = torch.log(z) - torch.log(1-z)

        return z, ldj

    def _dequantize_uniform(self, x, n_bins):
        """
        Rescales pixels and adds uniform noise for dequantization.
        """
        unif_noise = torch.zeros_like(x).uniform_(0, float(1.0 / n_bins))
        x = unif_noise + x

        # Initialize log determinant
        logdet = torch.zeros_like(x[:, 0, 0, 0, 0])

        # Log determinant adjusting for rescaling of 1/nbins for each pixel value
        logdet = logdet + float(-np.log(n_bins) * np.prod(x.size()[1:]))
        return x, logdet

    def _predict(self, x_past, state, eps=1.0):
        """
        Inference step for time-series forecasting.
        Obtain a sample of time-series for the next time step
        x_t1 = f^-1(z_{t1}|h_{t1}).
        """
        # sampling next time-step
        # TODO: later on, adapt code to sample trajectories
        # TODO: for trajectory sampling, can also use cold hidden start state with zeros
        with torch.no_grad():
            prediction = self.inverse_flow(x_past=x_past, state=state, eps=eps)
        return prediction
