from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MLGN.MLGN_model import MLGN_Generator, MLGN_Discriminator, LOAD_MLGN
import torch.nn.utils.spectral_norm as spectral_norm
from models.pSp.psp import LOAD_PSP_ENCODER


def build_MLGN_model(args):
    generator = MLGN_Generator()
    discriminator = MLGN_Discriminator(args)
    nets = Munch(generator=generator, discriminator=discriminator)
    return nets

def build_model(args):
    generator = InpaintingModel(args)
    discriminator = Discriminator(args)
    pretrained_MLGN = LOAD_MLGN(args)
    nets = Munch(generator=generator, discriminator=discriminator, MLGN=pretrained_MLGN)
    return nets

def build_psp_model(args):
    nets = Munch(PSP=LOAD_PSP_ENCODER(args))
    return nets

def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

class InpaintingModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.z_dim = args.z_dim
        self.start_h = args.start_h
        self.start_w = args.start_w
        self.ngf = args.ngf

        self.fc = nn.Linear(self.z_dim, 16 * self.ngf * self.start_h * self.start_w)

        self.head_0 = RN_SPD_ResnetBlock(16 * self.ngf, 16 * self.ngf)

        self.G_middle_0 = RN_SPD_ResnetBlock(16 * self.ngf, 16 * self.ngf)
        self.G_middle_1 = RN_SPD_ResnetBlock(16 * self.ngf, 16 * self.ngf)

        self.up_0 = RN_SPD_ResnetBlock(16 * self.ngf, 8 * self.ngf)
        self.up_1 = RN_SPD_ResnetBlock(8 * self.ngf, 4 * self.ngf)
        self.up_2 = RN_SPD_ResnetBlock(4 * self.ngf, 2 * self.ngf)
        self.up_3 = RN_SPD_ResnetBlock(2 * self.ngf, 1 * self.ngf)

        self.conv_img = nn.Conv2d(self.ngf, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

        self.conv0 = nn.ConvTranspose2d(16 * self.ngf, 16 * self.ngf, 4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(16 * self.ngf, 16 * self.ngf, 3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(16 * self.ngf, 16 * self.ngf, 4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(8 * self.ngf, 8 * self.ngf, 4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(4 * self.ngf, 4 * self.ngf, 4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(2 * self.ngf, 2 * self.ngf, 4, stride=2, padding=1)
        self.conv6 = nn.ConvTranspose2d(1 * self.ngf, 1 * self.ngf, 3, stride=1, padding=1)

        #self.constant_input = nn.Parameter(torch.randn(1, 18 * 512))

        # ENCODE SETTING
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = args.ngf
        norm_layer = get_nonspade_norm_layer(args, 'spectralinstance')
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.actvn = nn.LeakyReLU(0.2, False)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def forward(self, m_image, coarse_MLGN, mask):
        x = self.layer1(m_image)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.layer6(self.actvn(x))
        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        z = self.reparameterize(mu, logvar)

        x = self.fc(z)
        x = x.view(-1, 16 * self.ngf, self.start_h, self.start_w)

        x = self.head_0(x, coarse_MLGN, mask)
        x = self.conv0(x)

        x = self.G_middle_0(x, coarse_MLGN, mask)
        x = self.conv1(x)
        x = self.G_middle_1(x, coarse_MLGN, mask)
        x = self.conv2(x)

        x = self.up_0(x, coarse_MLGN, mask)
        x = self.conv3(x)

        x = self.up_1(x, coarse_MLGN, mask)
        x = self.conv4(x)

        x = self.up_2(x, coarse_MLGN, mask)
        x = self.conv5(x)

        x = self.up_3(x, coarse_MLGN, mask)
        x = self.conv6(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x, z

class RN_SPD_ResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.fin = fin
        self.fout = fout
        self.fmiddle = min(self.fin, self.fout)


        self.conv_0 = spectral_norm(nn.Conv2d(self.fin, self.fmiddle, kernel_size=3, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(self.fmiddle, self.fout, kernel_size=3, padding=1))

        self.conv_s = spectral_norm(nn.Conv2d(self.fin, self.fout, kernel_size=1, bias=False))

        self.norm_0 = RN_B((self.fin))
        self.norm_1 = RN_B((self.fmiddle))

        self.norm_s = RN_B((self.fin))

    def forward(self, x, coarse_image, mask):
        #soft
        x_s = self.conv_s(self.norm_s(x, coarse_image, mask))

        #hard
        dx = self.conv_0(self.actvn(self.norm_0(x, coarse_image, mask)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, coarse_image, mask)))

        out = x_s + dx
        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class RN_B(nn.Module):
    def __init__(self, norm_nc):
        super().__init__()
        self.rn = RN_binarylabel(norm_nc)
        nhidden = 128
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(3, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU())

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)


    def forward(self, x, coarse_image, mask):
        coarse_image_ = F.interpolate(coarse_image, size=x.size()[2:], mode='nearest')
        mask_ = F.interpolate(mask, size=x.size()[2:], mode='nearest')

        rn_x = self.rn(x, mask_)

        actv = self.mlp_shared(coarse_image_)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = rn_x * (1 + gamma) + beta

        return out

class RN_binarylabel(nn.Module):
    def __init__(self, feature_channels):
        super(RN_binarylabel, self).__init__()
        self.bn_norm = nn.BatchNorm2d(feature_channels, affine=False, track_running_stats=False)

    def forward(self, x, mask):
        '''
        input:  x: (B,C,M,N), features
                label: (B,1,M,N), 1 for foreground regions, 0 for background regions
        output: _x: (B,C,M,N)

        mask : masked region value is 1
        '''
        mask = mask.detach()

        rn_foreground_region = self.rn(x * mask, mask)

        rn_background_region = self.rn(x * (1 - mask), 1 - mask)

        return rn_foreground_region + rn_background_region

    def rn(self, region, mask):
        '''
        input:  region: (B,C,M,N), 0 for surroundings
                mask: (B,1,M,N), 1 for target region, 0 for surroundings
        output: rn_region: (B,C,M,N)
        '''
        shape = region.size()

        sum = torch.sum(region, dim=[0,2,3])  # (B, C) -> (C)
        Sr = torch.sum(mask, dim=[0,2,3])    # (B, 1) -> (1)
        Sr[Sr==0] = 1
        mu = (sum / Sr)     # (B, C) -> (C)

        return self.bn_norm(region + (1 - mask) * mu[None,:,None,None]) * (torch.sqrt(Sr / (shape[0] * shape[2] * shape[3])))[None,:,None,None]

#For feature matching loss
class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        inputChannels = 3
        self.args = args

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(inputChannels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, image):
        x1 = self.conv1(image)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        output = self.conv6(x5)

        return output, [x1, x2, x3, x4, x5]