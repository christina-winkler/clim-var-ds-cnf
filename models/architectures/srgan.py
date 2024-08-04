import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
# taken from: https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class GaussianPrior(nn.Module):
    def __init__(self, in_c, cond_channels):
        super(GaussianPrior, self).__init__()

        self.cond_channels = cond_channels
        self.conv = nn.Conv2d(in_c, 2, kernel_size=3, stride=1, padding=1)

    def final_prior(self, feat_map):
        h = self.conv(feat_map)
        mean, sigma = h[:, 0].unsqueeze(1), nn.functional.softplus(h[:, 1].unsqueeze(1).type(torch.DoubleTensor).cuda())
        return mean, sigma

    def forward(self, feat_map, eps=1.0):
        # sample from conditional prior
        mean, sigma = self.final_prior(feat_map)
        prior = torch.distributions.normal.Normal(loc=mean, scale=sigma*eps+0.00001)
        z = prior.sample().type(torch.FloatTensor).cuda()
        return z

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, height=None, width=None, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        # self.attn1 = SelfAttention(gc, height//2, width//2)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        # self.attn2 = SelfAttention(gc, height//2, width//2)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # x2a = self.attn1(x2).squeeze(2)
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # x4a = self.attn2(x4).squeeze(2)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, height=None, width=None, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc, height, width)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc, height, width)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc, height, width)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class Generator(nn.Module):
    """
    RDDB Net type Generator.
    """
    def __init__(self, in_nc, out_nc, height, width, nf, nb=16, s=2, gc=32):
        super(Generator, self).__init__()

        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc, height=height, width=width)

        self.s = s
        self.cond_prior = GaussianPrior(in_c=in_nc, cond_channels=128).cuda()

        self.conv_first = nn.Conv2d(2, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, n_layers=nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, eps=1.0):

        # sample z from conditional base density
        z = self.cond_prior(x, eps)

        # add residual connection
        za = torch.cat((x,z),1)

        fea = self.conv_first(za)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.relu(self.upconv1(F.interpolate(fea, scale_factor=self.s, mode='bicubic')))
        # fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.relu(self.HRconv(fea)))

        return out


class Discriminator(torch.nn.Module):
    def __init__(self, in_c, out_c, height, width):
        super(Discriminator, self).__init__()

        self.leak_value = 0.2
        self.bias = False
        self.height = height
        self.width = width

        self.f_dim = 32

        self.layer1 = self.conv_layer(1, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim, 2*self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)
        # self.attn1 = SelfAttention(2*self.f_dim, height, width)
        self.layer3 = self.conv_layer(2*self.f_dim, self.f_dim, kernel_size=3, stride=1, padding=1, bias=self.bias)
        # self.attn2 = SelfAttention(self.f_dim, height, width)
        self.layer4 = self.conv_layer(self.f_dim, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(height * width, 1),
            torch.nn.Sigmoid()
        )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False):
        layer = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm2d(output_dim),
            # torch.nn.LeakyReLU(self.leak_value, inplace=True)
            torch.nn.ReLU(True)
        )
        return layer

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(x.size(0),-1)
        out = self.layer5(out)
        return out.squeeze(1)

# class Discriminator(nn.Module):
#     def __init__(self, in_channels) -> None:
#         super(Discriminator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),

#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(512, 1024, kernel_size=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(1024, 1, kernel_size=1)
#         )

#     def forward(self, x):
#         batch_size = x.size(0)
#         return torch.sigmoid(self.net(x).view(batch_size))

class SelfAttention(nn.Module):
    def __init__(self, channels, height, width):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        bsz = x.shape[0]
        x = x.view(bsz, self.channels, self.height * self.width).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.height, self.width).unsqueeze(2)
