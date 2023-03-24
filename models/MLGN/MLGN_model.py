from munch import Munch
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from checkpoint import CheckPoint
from os.path import join as ospj

"""
Contains the implementation of generator described in MLGN.
Paper: https://ieeexplore.ieee.org/document/8784896
Pytorch implementation: https://github.com/JieLiu95/MLGN
"""

class MLGN_Generator(nn.Module):
    def __init__(self):
        super(MLGN_Generator, self).__init__()
        self.down_0_0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7, 2, 0, bias=True),
            nn.LeakyReLU(0.2)
        )
        # state size: 128 * 128

        self.down_0_1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(64, 128, 5, 2, 0, bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        # state size: 64 * 64

        self.down_0_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 192, 3, 2, 0, bias=True),
            nn.InstanceNorm2d(192),
            nn.LeakyReLU(0.2)
        )
        # state size: 32 * 32

        self.res_0_0 = nn.Sequential(
            ResnetBlock(192),
            ResnetBlock(192)
        )
        # state size: 32 * 32

        self.down_0_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(192, 256, 3, 2, 0, bias=True),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        # state size: 16 * 16

        # # # floor 0
        self.res_0_1 = nn.Sequential(
            ResnetBlock(256),
            ResnetBlock(256),
            ResnetBlock(256),
        )
        # state size: 16 * 16

        self.up_0_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv_0_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            ResnetBlock(128)
            # state size: 32 * 32
        )

        self.up_0_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv_0_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            ResnetBlock(64),
            # state size: 64 * 64
        )

        self.up_0_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv_0_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            ResnetBlock(64),
            ResnetBlock(64)
            # state size: 128 * 128
        )

        self.up_0_0 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv_0_0 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            # state size: 256 * 256
            nn.ReflectionPad2d(2),
            nn.Conv2d(64, 3, 5, 1),
            nn.Tanh()
            # state size: 256 * 256
        )

        # # # floor 1
        self.down_1_0 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(256, 128, 5, 2, 0, bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        # state size: 8 * 8

        self.conv_1_0 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.res_1_0 = nn.Sequential(
            ResnetBlock(128),
            ResnetBlock(128),
            ResnetBlock(128),
        )
        # state size: 8 * 8

        self.up_1_0 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv_1_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            ResnetBlock(128),
        )
        # state size: 16 * 16

        self.up_1_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv_1_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            ResnetBlock(64),
        )
        # state size: 32 * 32

        self.up_1_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv_1_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            ResnetBlock(64),
        )
        # state size: 64 * 64

        # # # floor 2
        self.down_2_0 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(128, 128, 5, 2, 0, bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        # state size: 4 * 4

        self.conv_2_0 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.res_2_0 = nn.Sequential(
            ResnetBlock(128),
            ResnetBlock(128),
            ResnetBlock(128),
        )
        # state size: 4 * 4

        self.up_2_0 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv_2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            ResnetBlock(128),
        )
        # state size: 8 * 8

        self.up_2_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv_2_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            ResnetBlock(128),
        )
        # state size: 16 * 16

        self.up_2_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv_2_3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            ResnetBlock(128),
        )
        # state size: 32 * 32

    def forward(self, input_data):
        out_0_0 = self.down_0_3(self.res_0_0(self.down_0_2(self.down_0_1(self.down_0_0(input_data)))))
        out_0_1 = self.conv_0_3(self.up_0_3(self.res_0_1(out_0_0)))

        out_1_0 = self.conv_1_0(self.down_1_0(out_0_0))
        out_1_1 = self.res_1_0(out_1_0)
        out_1 = self.conv_1_3(self.up_1_2(self.conv_1_2(self.up_1_1(self.conv_1_1(self.up_1_0(out_1_1))))))
        out_2_0 = self.res_2_0(self.conv_2_0(self.down_2_0(out_1_0)))
        out_2 = self.conv_2_3(self.up_2_2(self.conv_2_2(self.up_2_1(self.conv_2_1(self.up_2_0(out_2_0))))))
        out_0_2 = self.conv_0_2(self.up_0_2(torch.cat((out_0_1, out_2), 1)))
        out = self.conv_0_0(self.up_0_0(self.conv_0_1(self.up_0_1(torch.cat((out_0_2, out_1), 1)))))

        return out

class ResnetBlock(nn.Module):
    def __init__(self, channel):
        super(ResnetBlock, self).__init__()
        self.resnetblock = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3, 1, 0),
            nn.InstanceNorm2d(channel),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel, channel, 3, 1, 0),
            nn.InstanceNorm2d(channel)
        )

    def forward(self, input_data):
        out = self.resnetblock(input_data) + input_data
        return out

class MLGN_Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        inputChannels = 3
        self.args = args

        self.Conv = nn.Sequential(
            nn.Conv2d(inputChannels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2 , inplace=True),

            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fusionLayer = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, image):
        x = self.Conv(image)
        out = self.fusionLayer(x).view(image.size()[0], -1)
        return out

class LOAD_MLGN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nets = Munch(generator=MLGN_Generator())
        self.nets.generator.cuda()

        ckptio = CheckPoint(ospj(args.mlgn_checkpoint_dir, '{0:0>6}_models.ckpt'), **self.nets)
        ckptio.load(args.pretrained_mlgn_step)

    def forward(self, image, mask):
        m_image = torch.mul(image, mask)
        reverse_mask = 1. - mask # erased region value is 1
        semi_completion_image = self.nets.generator(torch.cat((m_image, reverse_mask), 1))
        completion_image = semi_completion_image * reverse_mask + image * mask
        return completion_image
