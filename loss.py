from munch import Munch
import torch.nn as nn
import torch
from torch import autograd
import pytorch_msssim
import numpy as np
import torch.nn.functional as F
import torchvision

def compute_G_loss(models, psp_model, args, image, m_image, mask, fac_weights, epoch):
    #erased region value is 1
    reverse_mask = 1. - mask

    with torch.no_grad():
        coarse_image = models.MLGN(image, mask)
        StyleGAN2_image, latent = psp_model.PSP(coarse_image)

        layers, boundaries, values = fac_weights
        boundary = boundaries[epoch % 19 + 1: epoch % 19 + 2]
        distances = np.linspace(args.start_distance, args.end_distance, args.style_sample_num)

    style_imgs = []
    comp_imgs = []
    for idx, distance in enumerate(distances):
        temp_code = latent.cpu().numpy().copy()
        temp_code[:, layers, :] += boundary * distance
        with torch.no_grad():
            tmp_style_img, style_latent = psp_model.PSP(coarse_image, layers, torch.tensor(temp_code).cuda())
        tmp_completion_image, _ = models.generator(m_image, tmp_style_img * reverse_mask + image * mask, reverse_mask)
        style_imgs.append(tmp_style_img)
        comp_imgs.append(tmp_completion_image)

    style_transfer_image = torch.stack((style_imgs), 0).view(-1,3,256,256).detach()
    completion_image = torch.stack((comp_imgs), 0).view(-1,3,256,256)

    image = image.repeat(args.style_sample_num, 1, 1, 1)
    mask = mask.repeat(args.style_sample_num, 1, 1, 1)
    reverse_mask = 1. - mask

    #WGAN-GP
    with torch.no_grad():
        g_out_fake, _ = models.discriminator(completion_image)
        _, g_real_feature = models.discriminator(image)
    adv_g_loss = g_out_fake.mean().sum()

    #HOLE & VALID LOSS
    criterionL1 = torch.nn.L1Loss()
    valid_loss = criterionL1(completion_image * mask, image * mask)
    hole_loss = criterionL1(completion_image * reverse_mask, image * reverse_mask)

    #perceptual loss
    criterionVGG = VGGLoss()
    prc_loss = criterionVGG(completion_image, style_transfer_image, 'perceptual', reverse_mask)

    #vgg stlye loss
    vgg_style_loss = criterionVGG(completion_image, style_transfer_image, 'vggstyle', reverse_mask)

    #MS-SSIM LOSS
    loss_ms_ssim = pytorch_msssim.MS_SSIM(data_range=1)
    if torch.cuda.is_available():
        loss_ms_ssim.cuda()
    loss_ms_ssim_value = (1 - loss_ms_ssim((completion_image + 1) * 0.5, (image + 1) * 0.5))

    g_loss = args.lambda_adv*adv_g_loss + \
             args.lambda_valid*valid_loss + \
             args.lambda_hole*hole_loss + \
             args.lambda_ssim*loss_ms_ssim_value + \
             args.lambda_prc*prc_loss + \
             args.lambda_style*vgg_style_loss

    return g_loss, Munch(g_adv=adv_g_loss, SSIM=loss_ms_ssim_value, hole=hole_loss, valid=valid_loss, precep=prc_loss, vgg_sty=vgg_style_loss)

def compute_D_loss(models, psp_model, args, image, m_image, mask, fac_weights, epoch):
    with torch.no_grad():
        coarse_image = models.MLGN(image, mask)
        StyleGAN2_image, latent = psp_model.PSP(coarse_image)

        layers, boundaries, _ = fac_weights
        boundary = boundaries[args.semantic_layer: args.semantic_layer + 1]
        distances = np.linspace(args.start_distance, args.end_distance, args.style_sample_num)

        style_imgs = []
        comp_imgs = []
        for idx, distance in enumerate(distances):
            temp_code = latent.cpu().numpy().copy()
            temp_code[:, layers, :] += boundary * distance
            tmp_style_img, style_latent = psp_model.PSP(coarse_image, layers, torch.tensor(temp_code).cuda())
            tmp_completion_image, _ = models.generator(m_image, tmp_style_img * (1. - mask) + image * mask, (1. - mask))
            style_imgs.append(tmp_style_img)
            comp_imgs.append(tmp_completion_image)
        completion_image = torch.stack((comp_imgs), 0).view(-1, 3, 256, 256)

    #WGAN-GP
    image.requires_grad_()
    image = image.repeat(args.style_sample_num, 1, 1, 1)
    d_out_real, _ = models.discriminator(image)
    d_out_fake, _ = models.discriminator(completion_image)

    adv_d_real_loss = d_out_real.mean().sum() * -1
    adv_d_fake_loss = d_out_fake.mean().sum()
    loss_gp = calc_gradient_penalty(models.discriminator, image, completion_image, args.lambda_gp, torch.cuda.is_available())

    adv_d_loss = -adv_d_real_loss + adv_d_fake_loss + args.lambda_gp*loss_gp

    return adv_d_loss, Munch(d_real=adv_d_real_loss, d_fake=adv_d_fake_loss, gp=loss_gp)

def calc_gradient_penalty(netD, real_data, fake_data, lambda_, cuda_):
    BATCH_SIZE = real_data.size()[0]
    DIM = real_data.size()[2]
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    if cuda_:
        alpha = alpha.cuda()

    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    if cuda_:
        interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda_ else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty.sum().mean()

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

# Perceptual loss that uses a pretrained VGG network (SPADE)
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y, type, mask=None):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0

        if type == 'perceptual':
            for i in range(len(x_vgg)):
                if mask is not None:
                    loss += self.weights[i] * self.criterion(F.interpolate(mask, size=x_vgg[i].size()[2:], mode='nearest').repeat(1, x_vgg[i].size(1), 1, 1) * x_vgg[i],
                                                             F.interpolate(mask, size=x_vgg[i].size()[2:], mode='nearest').repeat(1, x_vgg[i].size(1), 1, 1) * y_vgg[i].detach())
                else:
                    loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        if type == 'vggstyle':
            for i in range(len(x_vgg)):
                if mask is not None:
                    loss += self.weights[i] * self.criterion(gram_matrix(F.interpolate(mask, size=x_vgg[i].size()[2:], mode='nearest').repeat(1, x_vgg[i].size(1), 1, 1) * x_vgg[i]),
                                                             gram_matrix(F.interpolate(mask, size=x_vgg[i].size()[2:], mode='nearest').repeat(1, x_vgg[i].size(1), 1, 1) * y_vgg[i].detach()))
                else:
                    loss += self.weights[i] * self.criterion(gram_matrix(x_vgg[i]), gram_matrix(y_vgg[i].detach()))
        return loss

# VGG architecter, used for the perceptual loss using a pretrained VGG network (SPADE)
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out