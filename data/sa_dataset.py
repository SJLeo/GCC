from torchvision import transforms
import torch

from data.base_dataset import BaseDataset

import os
from PIL import Image

class SADataset(BaseDataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.data_folder = os.path.join(opt.dataroot, opt.phase)
        self.image_file_names = sorted(os.listdir(self.data_folder))
        self.batch_size = opt.batch_size
        self.z_dim = opt.z_dim
        self.imsize = opt.crop_size

        self.transform = self.get_transform(True, True, True, opt.center_crop)

    def get_transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize, self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def __getitem__(self, index):

        img = Image.open(os.path.join(self.data_folder, self.image_file_names[index]), mode='r')
        img = img.convert('RGB')
        trans_img = self.transform(img)
        images = {'z':torch.randn(self.z_dim), 'real_img': trans_img, 'img_path': self.image_file_names[index]}
        return images

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_file_names)
