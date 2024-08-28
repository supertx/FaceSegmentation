"""
@author supermantx
@date 2024/8/27 16:56
"""
import os

import torch
import cv2 as cv
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader, Dataset
from CMHQ_config import Config


class CelebAMaskHQ(Dataset):
    """
    CelebAMask-HQ dataset
    the dataset directory should be like:
    ----------------
    |CelebA-HQ-img
    |-- 0.jpg
    |-- 1.jpg
    |-- ...
    |CelebAMask-HQ-mask-anno
    |-- 0
    |   |-- 0_skin.png
    |   |-- 0_hair.png
    |   |-- ...
    |-- ...
    ----------------
    """

    def __init__(self, transform=None, cfg=None):
        super().__init__()
        self.img_root = os.path.join(cfg.dataset_root, cfg.img_root).__str__()
        self.mask_root = os.path.join(cfg.dataset_root, cfg.mask_root).__str__()

        self.imgs_path = []
        self.imgs_index = []
        self.__completeness_examine()
        self.transform = transform

        self.feas = cfg.feas
        self.fea_num = len(self.feas)
        self.cfg = cfg
        self.training = True
        self.colors = list(cfg.colors_rgb.values())

    def __completeness_examine(self):
        """
        search if there is a img that lack of mask file
        """
        for dirpath, dirnames, filenames in os.walk(self.img_root):
            if len(dirnames) > 0:
                continue
            file_len = len(filenames)
            for filename in filenames:
                index = int(filename.split(".")[0])
                if not os.path.exists(os.path.join(self.mask_root, f"{index:>05d}")):
                    print(f"lack of mask file for {filename}")
                else:
                    self.imgs_path.append(filename)
                    self.imgs_index.append(f"{index:>05d}")
            print(f"examine finished, total {file_len} files, {file_len - len(self.imgs_path)} mask files no exist")

    def __get_mask(self, img_index):
        masks = []
        for fea in self.feas:
            if os.path.isfile(os.path.join(self.mask_root, img_index, f"{img_index}_{fea}.png")):
                mask = cv.imread(os.path.join(self.mask_root, img_index, f"{img_index}_{fea}.png"))
                mask = mask[:, :, 0] / 255
                mask = torch.from_numpy(mask).unsqueeze(0)
                masks.append(mask)
            else:
                masks.append(torch.zeros(1, 112, 112))
        return torch.stack(masks, dim=0)

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = cv.imread(os.path.join(self.img_root, img_path))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = self.__get_mask(self.imgs_index[index])
        if not self.training:
            return img, mask

        if self.transform:
            img = self.transform(img)
        return img, mask

    def __len__(self) -> int:
        return len(self.imgs_path)

    def show(self, index):
        img, mask = self.__getitem__(index)
        if self.training:
            # reverse normalize
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            img = img * std + mean
            img = img.permute(1, 2, 0).detach().numpy()
        show_mask = np.zeros((112, 112, 3))
        for i in range(self.fea_num):
            show_mask *= (1.0 - mask[i].permute(1, 2, 0).detach().numpy())
            show_mask += mask[i].permute(1, 2, 0).detach().numpy() * self.colors[i]
        show_mask = show_mask.astype(np.uint8)
        plt.figure(figsize=(15, 5))
        subplot = plt.subplot(1, 3, 1)
        subplot.imshow(img)
        subplot.set_title("raw_img")
        subplot = plt.subplot(1, 3, 2)
        subplot.imshow(show_mask)
        subplot.set_title("mask")
        subplot = plt.subplot(1, 3, 3)
        subplot.imshow((img * 0.5 + show_mask * 0.5) / 255)
        subplot.set_title("result")
        plt.show()


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_data_loader(dataset_cfg, cfg=None):
    dataset = CelebAMaskHQ(transform=get_transform(), cfg=dataset_cfg)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)


if __name__ == '__main__':
    dataset = CelebAMaskHQ(cfg=Config())
    dataset.training = False
    dataset.show(999)
