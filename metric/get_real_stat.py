import argparse
import warnings

import numpy as np
import torch

from data import create_dataset
from metric.fid_score import _compute_statistics_of_ims
from metric.inception import InceptionV3
from utils import util


def main(opt):
    dataloader = create_dataset(opt)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids \
        else torch.device('cpu')
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx])
    inception_model.to(device)
    inception_model.eval()

    tensors = {}
    for i, data_i in enumerate(dataloader):
        if opt.dataset_mode == 'aligned':
            tensor = data_i['B' if opt.direction == 'AtoB' else 'A']
            tensors[data_i['B_paths' if opt.direction == 'AtoB' else 'A_paths'][0]] = tensor
        elif opt.dataset_mode == 'sa':
            tensor = data_i['real_img']
            tensors[data_i['img_path'][0]] = tensor
    tensors = torch.cat(list(tensors.values()), dim=0)
    tensors = util.tensor2imgs(tensors).astype(float)
    mu, sigma = _compute_statistics_of_ims(tensors, inception_model, 32, 2048, device, use_tqdm=True)
    np.savez(opt.output_path, mu=mu, sigma=sigma)

# CelebA: python3 metric/get_real_stat.py --dataroot ./database/celeb/ --dataset_mode sa --crop_size 64 --output_path ./database/real_stat.npz --gpu_ids 2 --center_crop
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract some statistical information of a dataset to compute FID')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--dataroot', required=True,
                        help='path to images (should have subfolders trainA, trainB, valA, valB, train, val, etc)')
    parser.add_argument('--dataset_mode', type=str, default='aligned',
                        help='chooses how datasets are loaded. [aligned | sr | sa]')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--preprocess', type=str, default='none', help='scaling and cropping of images at load time '
                             '[resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
    parser.add_argument('--output_path', type=str, required=True,
                        help='the path to save the statistical information.')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--z_dim', type=int, default=128, help='the noise input size. default:128')
    parser.add_argument('--center_crop', action='store_true')

    opt = parser.parse_args()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.max_dataset_size = -1
    opt.load_in_memory = False
    if opt.dataset_mode == 'single' and opt.direction == 'AtoB':
        warnings.warn('Dataset mode [single] only supports direction BtoA. '
                      'We will change the direction to BtoA.!')
        opt.direction = 'BtoA'


    def parse_gpu_ids(str_ids):
        str_ids = str_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
        return gpu_ids


    opt.gpu_ids = parse_gpu_ids(opt.gpu_ids)

    if not opt.output_path.endswith('.npz'):
        warnings.warn('The output is a numpy npz file, but the output path does\'nt end with ".npz".')
    if len(opt.gpu_ids) > 1:
        warnings.warn('The code only supports single GPU. Only gpu [%d] will be used.' % opt.gpu_ids[0])
    main(opt)
