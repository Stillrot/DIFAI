from pathlib import Path
from itertools import chain
import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from munch import Munch

class InputFetcher:
    def __init__(self, loader):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch_inputs(self):
        try:
            img, msk = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            img, msk = next(self.iter)
        return img, msk

    def __next__(self):
        img, msk = self._fetch_inputs()
        inputs = Munch(image=img, mask=msk)
        return Munch({k: v.to(self.device) for k, v in inputs.items()})

def listdir(dname):
    #fnames = list(chain(*[list(Path(dname).rglob('*.' + ext)) for ext in ['jpg']]))
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext)) for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

class dataset_loader(torch.utils.data.Dataset):
    def __init__(self, args, dl_mode):
        self.args = args
        self.img_size = self.args.img_size
        self.input_size = (self.img_size, self.img_size)
        self.padding = 50
        self.dl_mode = dl_mode

        self.all_images_name = np.loadtxt(self.args.image_list_dir, skiprows=2, usecols=[0], dtype=np.str)
        self.masks = self.load_masks(self.args.masks_dir)

        if self.dl_mode == 'train':
            self.image_dir = args.image_dir
            self.images_name = self.all_images_name[:28000]

        elif self.dl_mode == 'val':
            self.image_dir = args.image_val_dir
            self.images_name = self.all_images_name[28000:]

        else:
            print("Error")

        self.length = len(self.images_name)

    def load_masks(self, root):
        fnames = os.listdir(root)
        m_dir = []
        for fname in sorted(fnames):
            m_dir.append(os.path.join(root, fname))
        return m_dir

    def create_mask_box(self, width, height, mask_width, mask_height, x=None, y=None):
        mask = np.ones((height, width))
        mask_x = x if x is not None else random.randint(0 + self.padding, width - mask_width - self.padding)
        mask_y = y if y is not None else random.randint(0 + self.padding, height - mask_height - self.padding)
        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 0
        return mask

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_dir, self.images_name[index]))

        ld_mask = Image.open(self.masks[index])
        m_w, m_h = ld_mask.size
        mask = ld_mask * self.create_mask_box(m_w, m_h, m_w // 3, m_h // 3)
        mask = Image.fromarray(mask.astype('float32'))

        if self.dl_mode == 'train':
            self.img_transform = transforms.Compose([
                                    transforms.Resize(size=self.input_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])
                                    ])
            self.mask_transform = transforms.Compose([
                                    transforms.Resize(size = self.input_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()
                                    ])

        else:
            self.img_transform = transforms.Compose([
                                    transforms.Resize(size=self.input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])
                                    ])
            self.mask_transform = transforms.Compose([
                                    transforms.Resize(size=self.input_size),
                                    transforms.ToTensor()
                                    ])

        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return self.length