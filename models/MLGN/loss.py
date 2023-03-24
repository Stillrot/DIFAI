from munch import Munch
import torch.nn as nn
from torchvision import models
import torch
from torch import autograd
import pytorch_msssim

def compute_G_loss(models, args, image, m_image, mask, device):
    g_mask = 1. - mask
    m_image = torch.cat((m_image, g_mask), 1)

    semi_completion_image = models.generator(m_image)
    completion_image = semi_completion_image * g_mask + image * mask

    with torch.no_grad():
        cmp_D = models.discriminator(completion_image)

    # vgg style loss
    extractor = VGG16FeatureExtractor().to(device)
    feat_output_comp = extractor(completion_image)
    feat_output = extractor(semi_completion_image)
    feat_gt = extractor(image)

    l1 = nn.L1Loss()
    vgg_style_loss = 0.0
    for i in range(3):
        vgg_style_loss += l1(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i]))
        vgg_style_loss += l1(gram_matrix(feat_output_comp[i]), gram_matrix(feat_gt[i]))

    #MS-SSIM LOSS
    loss_ms_ssim = pytorch_msssim.MS_SSIM(data_range=1)
    if torch.cuda.is_available():
        loss_ms_ssim.cuda()
    loss_ms_ssim_value = (1 - loss_ms_ssim((completion_image + 1) * 0.5, (image + 1) * 0.5))

    # WGAN_GP: fake to real
    adv_fake_loss = cmp_D.mean().sum() * 1

    #HOLE LOSS
    criterionL1 = torch.nn.SmoothL1Loss()
    loss_G_L1 = criterionL1(completion_image * g_mask, image * g_mask)

    loss = 1 * vgg_style_loss + 5 * loss_ms_ssim_value + 0.1 * adv_fake_loss + 1 * loss_G_L1

    return loss, Munch(fake=adv_fake_loss, valid=loss_G_L1, MSSSIM=loss_ms_ssim_value, vgg_sty=vgg_style_loss)

def compute_D_loss(models, args, image, m_image, mask, device):
    g_mask = 1. - mask
    m_image = torch.cat((m_image, g_mask), 1)

    with torch.no_grad():
        semi_completion_image = models.generator(m_image)
        completion_image = semi_completion_image * g_mask + image * mask

    fake_D = models.discriminator(completion_image)

    image.requires_grad_()
    ori_D = models.discriminator(image)

    #WGAN-GP real to real
    adv_real_loss = ori_D.mean().sum() * -1
    #WGAN-GP fake to fake
    adv_fake_loss = fake_D.mean().sum() * 1

    loss_gp = calc_gradient_penalty(models.discriminator, image, completion_image, device, args.mlgn_lambda_gp)

    loss = adv_fake_loss - adv_real_loss + loss_gp
    return loss, Munch(real=adv_real_loss, fake=adv_fake_loss, gp=loss_gp)


def calc_gradient_penalty(netD, real_data, fake_data, device, lambda_):
    BATCH_SIZE = real_data.size()[0]
    DIM = real_data.size()[2]
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    alpha = alpha.to(device)

    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty.sum().mean()

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram