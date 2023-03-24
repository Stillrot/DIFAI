import os
from os.path import join as ospj
import torch
import torch.nn as nn
import torchvision.utils as vutils

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

@torch.no_grad()
def debug_image(models, args, sample_inputs, step):
    models = models
    image = sample_inputs.image
    mask = sample_inputs.mask

    if mask.size()[1] == 4 or args.mode == 'test':
        mask = mask[:, 0:1, :, :]

    N = image.size(0)
    if N > 8: N = 7

    m_image = torch.mul(image, mask)
    input_ = torch.cat((m_image, 1. - mask), 1)
    coarse_image = torch.mul(models.generator(input_), 1.-mask) + m_image

    filename1 = ospj(args.mlgn_val_dir, '%06d_1_input.jpg' % (step))
    save_image(m_image, N, filename1)

    filename2 = ospj(args.mlgn_val_dir, '%06d_2_coarse.jpg' % (step))
    save_image(coarse_image, N, filename2)