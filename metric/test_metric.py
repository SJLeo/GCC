import torch

from data import create_dataset
import utils.util as util
from metric import get_fid, get_mIoU
from metric.inception import InceptionV3
from metric.mIoU_score import DRNSeg

import ntpath
import os
import numpy as np
from tqdm import tqdm


def test_pix2pix_fid(model, opt):
    opt.phase = 'val'
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.load_size = 256
    dataset = create_dataset(opt)
    model.model_eval()

    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    util.mkdirs(result_dir)

    fake_B = {}
    for i, data in enumerate(dataset):
        model.set_input(data)
        with torch.no_grad():
            model.forward()
        visuals = model.get_current_visuals()
        fake_B[data['A_paths'][0]] = visuals['fake_B']
        # util.save_images(visuals, model.image_paths, result_dir, direction=opt.direction,
        #                  aspect_ratio=opt.aspect_ratio)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx])
    inception_model.to(model.device)
    inception_model.eval()
    npz = np.load(os.path.join(opt.dataroot, 'real_stat_B.npz' if opt.direction == 'AtoB' else 'real_stat_A.npz'))
    fid = get_fid(list(fake_B.values()), inception_model, npz, model.device, opt.batch_size)

    return fid

def test_pix2pix_mIoU(model, opt):
    opt.phase = 'val'
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.load_size = 256
    dataset = create_dataset(opt)
    model.model_eval()

    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    util.mkdirs(result_dir)

    fake_B = {}
    names = []
    for i, data in enumerate(dataset):
        model.set_input(data)

        with torch.no_grad():
            model.forward()
        visuals = model.get_current_visuals()
        fake_B[data['A_paths'][0]] = visuals['fake_B']

        for path in range(len(model.image_paths)):
            short_path = ntpath.basename(model.image_paths[0][0])
            name = os.path.splitext(short_path)[0]
            if name not in names:
                names.append(name)
        util.save_images(visuals, model.image_paths, result_dir, direction=opt.direction,
                         aspect_ratio=opt.aspect_ratio)

    drn_model = DRNSeg('drn_d_105', 19, pretrained=False).to(model.device)
    util.load_network(drn_model, opt.drn_path, verbose=False)
    drn_model.eval()

    mIoU = get_mIoU(list(fake_B.values()), names, drn_model, model.device,
                    table_path=os.path.join(opt.dataroot, 'table.txt'),
                    data_dir=opt.dataroot,
                    batch_size=opt.batch_size,
                    num_workers=opt.num_threads)
    return mIoU

def test_srgan_psnr(model, opt, dataset_name='Set5'):

    opt.phase = 'test/' + dataset_name
    opt.batch_size = 1
    opt.serial_batches = True
    dataset = create_dataset(opt)
    model.model_eval()

    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results', dataset_name)
    util.mkdirs(result_dir)

    # names = []
    psnrs = []
    ssims = []
    for i, data in enumerate(tqdm(dataset)):
        model.set_input(data)

        with torch.no_grad():
            model.forward()

        psnr = model.get_current_psnr()
        ssim = model.get_current_ssim()
        psnrs.append(psnr)
        ssims.append(ssim)

        # visuals = model.get_current_visuals()
        #
        # for path in range(len(model.image_paths)):
        #     short_path = ntpath.basename(model.image_paths[0][0])
        #     name = os.path.splitext(short_path)[0]
        #     if name not in names:
        #         names.append(name)
        # util.save_images(visuals, model.image_paths, result_dir, direction=opt.direction,
        #                  aspect_ratio=opt.aspect_ratio)

    # psnr = util.compute_tensor_psnr(fake_hr, real_hr)
    # psnr = util.compute_psnr(result_dir, os.path.join(opt.dataroot, 'val', dataset_name, 'hr'))

    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)

def test_sagan_fid(model, opt):

    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.load_size = 64
    dataset = create_dataset(opt)
    model.model_eval()

    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    util.mkdirs(result_dir)

    fake_img = {}
    for i, data in enumerate(dataset):

        if i > len(dataset) * 0.1:
            break
        model.set_input(data)
        with torch.no_grad():
            model.forward()
        visuals = model.get_current_visuals()
        fake_img[data['img_path'][0]] = visuals['fake_img']
        # util.save_images(visuals, model.image_paths, result_dir, direction=opt.direction,
        #                  aspect_ratio=opt.aspect_ratio)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx])
    inception_model.to(model.device)
    inception_model.eval()
    npz = np.load(os.path.join(opt.dataroot, 'real_stat.npz'))
    fid = get_fid(list(fake_img.values()), inception_model, npz, model.device, opt.batch_size)

    return fid

def test_cyclegan_fid(model, opt):
    opt.phase = 'test'
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.load_size = 256
    dataset = create_dataset(opt)
    model.model_eval()

    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    util.mkdirs(result_dir)

    fake_A = {}
    fake_B = {}

    for i, data in enumerate(dataset):
        model.set_input(data)
        with torch.no_grad():
            model.forward()
        visuals = model.get_current_visuals()
        fake_B[data['A_paths'][0]] = visuals['fake_B']
        fake_A[data['B_paths'][0]] = visuals['fake_A']
        # util.save_images(visuals, model.image_paths, result_dir, direction=opt.direction,
        #                  aspect_ratio=opt.aspect_ratio)

    # print('Calculating AtoB FID...', flush=True)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx])
    inception_model.to(model.device)
    inception_model.eval()
    npz = np.load(os.path.join(opt.dataroot, 'real_stat_B.npz'))
    AtoB_fid = get_fid(list(fake_B.values()), inception_model, npz, model.device, opt.batch_size)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx])
    inception_model.to(model.device)
    inception_model.eval()
    npz = np.load(os.path.join(opt.dataroot, 'real_stat_A.npz'))
    BtoA_fid = get_fid(list(fake_A.values()), inception_model, npz, model.device, opt.batch_size)

    return AtoB_fid, BtoA_fid