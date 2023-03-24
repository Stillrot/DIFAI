import os
from os.path import join as ospj
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from models.styleGAN2 import stylegan2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

def parse_indices(obj, min_val=None, max_val=None):
    if obj is None or obj == '':
        indices = []
    elif isinstance(obj, int):
        indices = [obj]
    elif isinstance(obj, (list, tuple, np.ndarray)):
        indices = list(obj)
    elif isinstance(obj, str):
        indices = []
        splits = obj.replace(' ', '').split(',')
        for split in splits:
            numbers = list(map(int, split.split('-')))
            if len(numbers) == 1:
                indices.append(numbers[0])
            elif len(numbers) == 2:
                indices.extend(list(range(numbers[0], numbers[1] + 1)))
            else:
                raise ValueError(f'Unable to parse the input!')

    else:
        raise ValueError(f'Invalid type of input: `{type(obj)}`!')

    assert isinstance(indices, list)
    indices = sorted(list(set(indices)))
    for idx in indices:
        assert isinstance(idx, int)
        if min_val is not None:
            assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
        if max_val is not None:
            assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

    return indices

def factorize_weight(generator, layer_idx='all'):
    # Get GAN type.
    gan_type = 'stylegan2'

    # Get layers.
    if gan_type == 'pggan':
        layers = [0]
    elif gan_type in ['stylegan', 'stylegan2']:
        if layer_idx == 'all':
            layers = list(range(generator.num_layers))
        else:
            layers = parse_indices(layer_idx,
                                   min_val=0,
                                   max_val=generator.num_layers - 1)

    # Factorize semantics from weight.
    weights = []
    for idx in layers:
        layer_name = f'layer{idx}'
        if gan_type == 'stylegan2' and idx == generator.num_layers - 1:
            layer_name = f'output{idx // 2}'
        if gan_type == 'pggan':
            weight = generator.__getattr__(layer_name).weight
            weight = weight.flip(2, 3).permute(1, 0, 2, 3).flatten(1)
        elif gan_type in ['stylegan', 'stylegan2']:
            weight = generator.synthesis.__getattr__(layer_name).style.weight.T
        weights.append(weight.cpu().detach().numpy())
    weight = np.concatenate(weights, axis=1).astype(np.float32)
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))

    return layers, eigen_vectors.T, eigen_values

def load_stylegan(args):
    generator = stylegan2.StyleGAN2Generator(resolution=1024)
    checkpoint = torch.load(args.stylegan2_checkpoint_path, map_location='cpu')
    generator.load_state_dict(checkpoint['generator'])
    generator = generator.to(args.device)
    generator.eval()
    return generator

@torch.no_grad()
def debug_image(models, psp_model, args, sample_inputs, step, img_name=None):
    image = sample_inputs.image
    mask = sample_inputs.mask
    m_image = torch.mul(image, mask)

    if mask.size()[1] >= 3: mask = mask[:, 0:1, :, :]
    N = image.size(0)
    if N > 8: N = 7

    reverse_mask = 1. - mask
    coarse_image = models.MLGN(image, mask)
    StyleGAN2_image, latent = psp_model.PSP(coarse_image)
    coarse_completion_image, z_ = models.generator(m_image, StyleGAN2_image * reverse_mask + image * mask, reverse_mask)

    layers, boundaries, values = factorize_weight(load_stylegan(args), args.layer_idx)
    boundary = boundaries[step % 19 + 1: step % 19 + 2]
    distances = np.linspace(args.start_distance, args.end_distance, args.style_sample_num)

    comp_imgs = []
    for idx, distance in enumerate(distances):
        temp_code = latent.cpu().numpy().copy()
        temp_code[:, layers, :] += boundary * distance
        tmp_style_img, style_latent = psp_model.PSP(coarse_image, layers, torch.tensor(temp_code).to(args.device))
        tmp_completion_image, _ = models.generator(m_image, tmp_style_img * reverse_mask + image * mask, reverse_mask)
        comp_imgs.append(tmp_completion_image)

    comp_imgs = torch.stack(comp_imgs)

    if args.mode == 'test':
        filename1 = ospj(args.test_sample_dir, '1_input_%s.jpg' % (img_name))
    elif args.mode == 'val':
        filename0 = ospj(args.val_sample_dir, '%06d_0_original.jpg' % (step))
        save_image(image, N+1, filename0)
        filename1 = ospj(args.val_sample_dir, '%06d_1_input.jpg' % (step))
    #elif args.mode == 'train':
    else:
        raise NotImplementedError
    save_image(m_image, N+1, filename1)

    if args.mode == 'test':
        filename2 = ospj(args.test_sample_dir, '2_coarse_%s.jpg' % (img_name))
    elif args.mode == 'val':
        filename2 = ospj(args.val_sample_dir, '%06d_2_coarse.jpg' % (step))
    #elif args.mode == 'train':
    else:
        raise NotImplementedError
    save_image(coarse_image, N+1, filename2)

    if args.mode == 'test':
        filename3 = ospj(args.test_sample_dir, '3_StyleGAN2_%s.jpg' % (img_name))
    elif args.mode == 'val':
        filename3 = ospj(args.val_sample_dir, '%06d_3_StyleGAN2.jpg' % (step))
    #elif args.mode == 'train':
    else:
        raise NotImplementedError
    save_image(StyleGAN2_image, N+1, filename3)

    if args.mode == 'test':
        filename4 = ospj(args.test_sample_dir, '4_DIFAI_input_%s.jpg' % (img_name))
    elif args.mode == 'val':
        filename4 = ospj(args.val_sample_dir, '%06d_4_DIFAI_input.jpg' % (step))
    #elif args.mode == 'train':
    else:
        raise NotImplementedError
    save_image(StyleGAN2_image*reverse_mask + image*mask, N+1, filename4)

    if args.mode == 'test':
        filename5 = ospj(args.test_sample_dir, '5_completion_vanila_%s.jpg' % (img_name))
    elif args.mode == 'val':
        filename5 = ospj(args.val_sample_dir, '%06d_5_completion_vanila.jpg' % (step))
    #elif args.mode == 'train':
    else:
        raise NotImplementedError
    save_image(coarse_completion_image, N+1, filename5)

    for idx, style_comp in enumerate(comp_imgs):
        if args.mode == 'test':
            filename6 = ospj(args.test_sample_dir, '6_completion_style_%s_%02d.jpg' % (img_name, idx))
        elif args.mode == 'val':
            filename6 = ospj(args.val_sample_dir, '%06d_6_completion_style_%02d.jpg' % (step, idx))
        # elif args.mode == 'train':
        else:
            raise NotImplementedError
        save_image(style_comp, N + 1, filename6)