from options import options
from models import Pix2Pix
import utils.util as util

import torch.nn as nn

import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    opt = options.parse()
    opt.isTrain = True
    safe_path = os.path.join(opt.checkpoints_dir, opt.name, 'weight_distribute')
    util.mkdirs(safe_path)

    # create model
    if opt.model == 'pix2pix':
        opt.norm = 'batch'
        opt.dataset_mode = 'aligned'
        opt.pool_size = 0
        model = Pix2Pix.Pix2PixModel(opt)
    else:
        raise NotImplementedError('%s not implemented' % opt.model)

    model.load_models(opt.pretrain_path)

    for name, module in model.netG.named_modules():

        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            print(name)
            plt.figure()
            plt.title(name)
            plt.hist(module.weight.data.view(-1).cpu(), 50, density=True, facecolor='g', alpha=0.75)
            plt.savefig(os.path.join(safe_path, '%s.png' % name))
            plt.close()


