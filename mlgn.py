import os
from os.path import join as ospj
from munch import Munch
import time
import random
import datetime
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import network
from torchvision import transforms
from checkpoint import CheckPoint
from PIL import Image
from models.MLGN.utils import he_init, debug_image, save_image
import dataloader as dl
from models.MLGN.loss import compute_D_loss, compute_G_loss

class MLGN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.models = network.build_MLGN_model(args)

        self.models.generator.cuda()
        self.models.discriminator.cuda()

        self.optims = Munch()
        for model in self.models.keys():
            self.optims[model] = torch.optim.Adam(params=self.models[model].parameters(),
                                                  lr=args.d_lr if 'discriminator' in model else args.g_lr,
                                                  betas=[args.beta1, args.beta2] if 'discriminator' in model else [0.5, 0.9])

        self.ckptios = [
            CheckPoint(ospj(args.mlgn_checkpoint_dir, '{0:0>6}_models.ckpt'), **self.models),
            CheckPoint(ospj(args.mlgn_checkpoint_dir, '{0:0>6}_optims.ckpt'), **self.optims)
            ]

        for name, model in self.models.items():
            print('Initializing MLGN %s...' % name)
            model.apply(he_init)

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
        fetcher = dl.InputFetcher(train_loader, 'train')

        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        self.models.generator.train()
        self.models.discriminator.train()

        start_time = time.time()
        for epoch in range(args.resume_iter, args.total_iters):
            inputs = next(fetcher)
            image = inputs.image
            mask = inputs.mask

            m_image = torch.mul(image, mask)

            # D train
            d_loss, d_loss_group = compute_D_loss(self.models, self.args, image, m_image, mask, self.device)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # G train
            g_loss, g_loss_group = compute_G_loss(self.models, self.args, image, m_image, mask, self.device)
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
                self._save_checkpoint(step=epoch + 1)

    @torch.no_grad()
    def val(self, args):
        # only validate inpainting quality
        val_models = self.models
        self._load_checkpoint(args.resume_iter)
        vl_loader = dl.dataset_loader(args, 'val')
        val_loader = DataLoader(dataset=vl_loader, batch_size=args.batch_size, num_workers=4, shuffle=False)
        fetcher = dl.InputFetcher(val_loader)

        tmp = random.randint(0, 1992 // args.batch_size)
        for _ in range(tmp):
            _ = next(fetcher)
        inputs = next(fetcher)

        os.makedirs(args.mlgn_val_dir, exist_ok=True)
        print('Working on {}...'.format(ospj(args.mlgn_val_dir, 'mlgn_validation.jpg')))
        debug_image(val_models, args, inputs, args.resume_iter)

    @torch.no_grad()
    def test(self, args):
        test_models = self.models
        os.makedirs(args.test_sample_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        images_name = os.listdir(args.image_test_dir)
        for image_name in images_name:
            name = str(image_name.split('.')[0])
            src_img = Image.open(os.path.join(args.image_test_dir, image_name))
            mask = Image.open(os.path.join(args.masks_test_dir, image_name))

            img_transform = transforms.Compose([transforms.Resize(size=args.img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            mask_transform = transforms.Compose([transforms.Resize(size=args.img_size),
                                                 transforms.Grayscale(num_output_channels=1),
                                                 transforms.ToTensor()])

            src_img = img_transform(src_img).to(self.device)
            mask = mask_transform(mask).to(self.device)
            m_image = torch.mul(src_img, mask)

            comp_image = test_models.generator(torch.cat((m_image, 1.-mask), 0).unsqueeze(0))
            completion_image = torch.mul(comp_image, 1.-mask) + torch.mul(src_img, mask)

            filename = ospj(args.test_sample_dir, 'MLGN_test_{0}.jpg'.format(name))
            save_image(completion_image, 1, filename)
