import os
from os.path import join as ospj
import random
from PIL import Image
from munch import Munch
import time
import datetime
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import factorize_weight, load_stylegan
import network as CM
from checkpoint import CheckPoint
import utils
import dataloader as dl
from loss import compute_D_loss, compute_G_loss

class DIFAI(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.models = CM.build_model(args)

        self.models.generator.to(self.args.device)
        self.models.discriminator.to(self.args.device)
        self.models.MLGN.to(self.args.device)

        self.psp_model = CM.build_psp_model(args)
        self.optims = Munch()

        model_names = ['generator', 'discriminator']

        for model in model_names:
            self.optims[model] = torch.optim.Adam(params=self.models[model].parameters(),
                lr=args.d_lr if 'discriminator' in model else args.g_lr,
                betas=[args.beta1, args.beta2] if 'discriminator' in model else [0.0, 0.99])

        self.ckptios = [
            CheckPoint(ospj(args.checkpoint_dir, '{0:0>6}_models.ckpt'), **self.models),
            CheckPoint(ospj(args.checkpoint_dir, '{0:0>6}_optims.ckpt'), **self.optims)
            ]

        for name, model in self.models.items():
            print('Initializing %s...' % name)
            model.apply(utils.he_init)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def train(self, args):
        optims = self.optims

        print('train dataloader')
        tr_loader = dl.dataset_loader(args, 'train')
        train_loader = DataLoader(dataset=tr_loader, batch_size=args.batch_size, num_workers=4, shuffle=True)
        fetcher = dl.InputFetcher(train_loader)

        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        self.models.generator.train()
        self.models.discriminator.train()

        start_time = time.time()
        fac_weights = factorize_weight(load_stylegan(args), args.layer_idx)

        for epoch in range(args.resume_iter, args.total_iters):
            inputs = next(fetcher)
            image = inputs.image
            mask = inputs.mask
            m_image = torch.mul(image, mask)

            # D train
            d_loss, d_loss_group = compute_D_loss(self.models, self.psp_model, self.args, image, m_image, mask, fac_weights, epoch)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # G train
            g_loss, g_loss_group = compute_G_loss(self.models, self.psp_model, self.args, image, m_image, mask, fac_weights, epoch)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            if (epoch + 1) % args.verbose_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, epoch + 1, args.total_iters)

                all_losses = dict()
                for loss, prefix in zip([d_loss_group, g_loss_group], ['  D/_', '  G/_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value

                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            if (epoch + 1) % args.save_step == 0:
                self._save_checkpoint(step = epoch + 1)

    @torch.no_grad()
    def val(self, args):
        val_models = self.models
        val_psp_models = CM.build_psp_model(args)
        os.makedirs(args.val_sample_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)
        v_loader = dl.dataset_loader(args, 'val')
        val_loader = DataLoader(dataset=v_loader, batch_size=args.batch_size, num_workers=4, shuffle=True)
        fetcher = dl.InputFetcher(val_loader)

        tmp = random.randint(0, 2000 // args.batch_size)
        for _ in range(tmp):
            _ = next(fetcher)
        inputs = next(fetcher)

        utils.debug_image(val_models, val_psp_models, args, sample_inputs=inputs, step=args.resume_iter)

    @torch.no_grad()
    def test(self, args):
        test_models = self.models
        test_psp_models = CM.build_psp_model(args)
        os.makedirs(args.test_sample_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)
        images_name = os.listdir(args.image_test_dir)

        img_transform = transforms.Compose([transforms.Resize(size=args.img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        mask_transform = transforms.Compose([transforms.Resize(size=args.img_size),
                                            transforms.ToTensor()])

        for image_name in images_name:
            src_img = Image.open(ospj(args.image_test_dir, image_name))
            mask_ = Image.open(ospj(args.masks_test_dir, image_name))

            src_img = img_transform(src_img).unsqueeze(0).to(self.args.device)
            mask_ = mask_transform(mask_).unsqueeze(0).to(self.args.device)

            utils.debug_image(test_models,
                              test_psp_models,
                              args,
                              sample_inputs=Munch(image=src_img, mask=mask_),
                              step=args.resume_iter,
                              img_name=image_name.split('.')[0])