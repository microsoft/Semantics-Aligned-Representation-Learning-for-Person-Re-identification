from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import io

import torch
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

from torchvision.transforms import *
class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.totensor = ToTensor()
        self.normalize = Normalize([.5, .5, .5], [.5, .5, .5])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        # Add by Xin Jin, for getting texture:
        img_texture = read_image(img_path.replace('images_labeled', 'texture_cuhk03_labeled'))
        
        if self.transform is not None:
            img = self.transform(img)
            img_texture = self.normalize(self.totensor(img_texture))
        
        return img, pid, camid, img_path, img_texture