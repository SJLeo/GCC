import torch

from options import options
from data import create_dataset
from models import Pix2Pix, SAGAN, SRGAN, CycleGAN
import utils.util as util

import os
import copy
from tqdm import tqdm

def test_pix2pix_mIoU(model, opt):
    opt.phase = 'val'
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.load_size = 256
    dataset = create_dataset(opt)
    model.model_eval()

    display_images = []
    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    util.mkdirs(result_dir)
    for i, data in enumerate(dataset):

        # if i == 100:
        #     break
        model.set_input(data)
        with torch.no_grad():
            model.forward()

        current_visual_result = model.get_current_visuals()
        util.save_images(current_visual_result, model.image_paths, result_dir, direction=opt.direction,
                         aspect_ratio=opt.aspect_ratio)

def test_sagan_fid(model, opt):

    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.load_size = 64
    dataset = create_dataset(opt)
    model.model_eval()

    display_images = []
    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    util.mkdirs(result_dir)
    for i, data in enumerate(dataset):

        if i == 1000:
            break
        model.set_input(data)
        with torch.no_grad():
            model.forward()
        current_visual_result = model.get_current_visuals()
        util.save_images(current_visual_result, model.image_paths, result_dir, direction=opt.direction,
                         aspect_ratio=opt.aspect_ratio)

def test_srgan_psnr(model, opt):


    model.model_eval()
    test_dataset_names = ['Set5', 'Set14', 'B100', 'Urban100']

    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    util.mkdirs(os.path.join(result_dir, 'Set5'))
    util.mkdirs(os.path.join(result_dir, 'Set14'))
    util.mkdirs(os.path.join(result_dir, 'B100'))
    util.mkdirs(os.path.join(result_dir, 'Urban100'))


    for dataset_name in test_dataset_names:
        opt.phase = 'test/' + dataset_name
        opt.batch_size = 1
        opt.serial_batches = True
        dataset = create_dataset(opt)
        for i, data in enumerate(tqdm(dataset)):

            model.set_input(data)
            with torch.no_grad():
                model.forward()

            current_visual_result = model.get_current_visuals()
            util.save_images(current_visual_result, model.image_paths, os.path.join(result_dir, dataset_name), direction=opt.direction,
                             aspect_ratio=opt.aspect_ratio)

def test_cyclegan_fid(model, opt):
    opt.phase = 'test'
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.load_size = 256
    dataset = create_dataset(opt)
    model.visual_names = ['real_A', 'fake_B']
    model.model_eval()

    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    util.mkdirs(result_dir)


    for i, data in enumerate(dataset):
        model.set_input(data)
        with torch.no_grad():
            model.visual_forward()
        visuals = model.get_current_visuals()

        util.save_images(visuals, model.image_paths, result_dir, direction=opt.direction,
                         aspect_ratio=opt.aspect_ratio)


def test(model, opt):

    if opt.model == 'pix2pix':

        test_pix2pix_mIoU(model, copy.deepcopy(opt))

    elif opt.model == 'sagan':

        test_sagan_fid(model, copy.deepcopy(opt))

    elif opt.model == 'srgan':

        test_srgan_psnr(model, copy.deepcopy(opt))

    elif opt.model == 'cyclegan':

        test_cyclegan_fid(model, copy.deepcopy(opt))

if __name__ == '__main__':

    opt = options.parse()
    opt.isTrain = True
    util.mkdirs(os.path.join(opt.checkpoints_dir, opt.name))

    device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
    if not os.path.exists(opt.pretrain_path):
        raise FileNotFoundError('pretrain model path must be exist!!!')
    ckpt = torch.load(opt.pretrain_path, map_location=device)
    filter_cfgs, channel_cfgs = ckpt['cfg']

    if opt.model == 'pix2pix':
        # create model
        model = Pix2Pix.Pix2PixModel(opt, filter_cfgs=filter_cfgs, channel_cfgs=channel_cfgs)
    elif opt.model == 'srgan':
        model = SRGAN.SRGAN(opt, filter_cfgs=filter_cfgs)
    elif opt.model == 'sagan':
        model = SAGAN.SAGANModel(opt, filter_cfgs=filter_cfgs)
    elif opt.model == 'cyclegan':
        model = CycleGAN.MobileCycleGANModel(opt, cfg_AtoB=filter_cfgs, cfg_BtoA=channel_cfgs)
    else:
        raise NotImplementedError('%s not implemented' % opt.model)

    model.load_models(opt.pretrain_path, load_discriminator=False)

    # create dataset
    dataset = create_dataset(opt)

    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    test(model, copy.deepcopy(opt))