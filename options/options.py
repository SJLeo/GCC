import argparse
import os

import utils.util as util

parser = argparse.ArgumentParser('GAN-Compression')

# basic parameters
parser.add_argument('--dataroot', #required=True,
                    help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--name', type=str, default='default', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir', type=str, default='./experiments', help='models are saved here')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc. default:train')
parser.add_argument('--load_path', type=str, default=None, help='The path of load model. default:None')
parser.add_argument('--pretrain_path', type=str, default=None, help='The path of pretrain model. defalut:None')

# model parameters
parser.add_argument('--model', type=str, default='pix2pix',
                    help='chooses which model to use. [cyclegan | pix2pix]. default:pix2pix')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale. default:3')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale. default:3')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer. default:64.')
parser.add_argument('--pretrain_ngf', type=int, default=64, help='# of teacher gen filters in the last conv layer. default:64.')
parser.add_argument('--ndf', type=int, default=128, help='# of discrim filters in the first conv layer. default:64.')
parser.add_argument('--backbone', type=str, default='unet', help='the backbone of generator. [unet | resnet] default:unet.')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--num_downs', type=int, default=8)
parser.add_argument('--continue_train', type=bool, default=False, help='continue training: load the latest model')

# dataset parameters
parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [sr | aligned]')
parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data. default:8')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size. default:1')
parser.add_argument('--load_size', type=int, default=286, help='scale images to this size. default:286')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size. default:256')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
# parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
parser.add_argument('--split_dataset', action='store_true', help='split train datasets into train/val')

# train parameter
# parser.add_argument('--display_freq', type=int, default=500, help='frequency of showing training results on screen')
# parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
# parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
# parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
# parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
# parser.add_argument('--display_port', type=int, default=9001, help='visdom port of the web display')
# parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
parser.add_argument('--print_freq', type=int, default=500, help='frequency of showing training results on console')
# parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs. default:1')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ... default:1')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate. default:100')
parser.add_argument('--n_epochs_decay', type=int, default=150, help='number of epochs to linearly decay learning rate to zero. default:150')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam. default:0.0002')
parser.add_argument('--gan_mode', type=str, default='hinge', help='the type of GAN objective. [vanilla| lsgan | hinge | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--pool_size', type=int, default=100, help='the size of image buffer that stores previously generated images. default:100')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A) default:10.0')
parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B) default:10.0')
parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1. default:0.5')
parser.add_argument('--lambda_L1', type=float, default=0.0, help='weight for L1 loss. default:0.0')

#test parameter
parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
parser.add_argument('--drn_path', type=str, default='./database/cityscapes/drn-d-105_ms_cityscapes.pth', help='the path of drm model for mAP computation. default:~/pretrain/drn-d-105_ms_cityscapes.pth')

# prune parameter
parser.add_argument('--scale_prune', action='store_true')
parser.add_argument('--norm_prune', action='store_true')
parser.add_argument('--lambda_weight', type=float, default=0.0, help='weight for weight L1 loss. default:0.0')
parser.add_argument('--lambda_scale', type=float, default=0.0, help='weight for BN L1 loss. default:0.0')
# parser.add_argument('--hdfs_save_path', type=str, default='hdfs://haruna/home/byte_labcv_default/user/lishaojie/GAN-Compression/')
parser.add_argument('--target_budget', type=float, default=None, help='the target budget of macs, the unit is G or G_A')
parser.add_argument('--target_budget_B', type=float, default=None, help='the target budget of macs, the unit is G_B')

# parser.add_argument('--ndf_scale', type=float, default=1.0)
parser.add_argument('--lottery_path', type=str, help='the path of initial model for lottery theory.')

# darts parametes
parser.add_argument('--darts_discriminator', action='store_true', help='using darts in discriminator')
parser.add_argument('--arch_lr', type=float, default=1e-4, help='initial learning rate for adam. default:1e-4')
parser.add_argument('--arch_lr_step', action='store_true')
parser.add_argument('--lambda_alpha', type=float, default=0.01, help='weight for arch loss. default:0.01')
parser.add_argument('--ema_beta', type=float, default=1.0, help='beta for ema. default:1.0')
parser.add_argument('--adaptive_ema', action='store_true')
parser.add_argument('--regular', action='store_true', help='regular the discriminator arch loss')
parser.add_argument('--arch_base_loss', action='store_true', help='the discriminator loss')
parser.add_argument('--only_arch_base', action='store_true', help='the discriminator loss')
parser.add_argument('--normalize_arch', action='store_true')
parser.add_argument('--clear_arch', action='store_true')
parser.add_argument('--threshold', type=float, default=0.5)

# dillation
parser.add_argument('--online_distillation', action='store_true')
parser.add_argument('--normal_distillation', action='store_true')
parser.add_argument('--distillation_path', type=str, default=None, help='the path of model for normal distillation')
parser.add_argument('--lambda_content', type=float, default=0.0, help='weight for content loss. default:0.0')
parser.add_argument('--lambda_gram', type=float, default=0.0, help='weight for gram loss. default:0.0')
parser.add_argument('--teacher_ngf', type=int, default=64, help='teacher ngf for distillation. default:64')
parser.add_argument('--teacher_ndf', type=int, default=64, help='teacher ndf for distillation. default:64')


# super-resolution
parser.add_argument('--lambda_SR_adversarial', type=float, default=1e-3, help='adversarial loss factor when '
                                                                                  'training generator oriented')
parser.add_argument('--lambda_SR_content', type=float, default=0.0, help='content loss factor when '
                                                                                  'training generator oriented')
parser.add_argument('--lambda_SR_perceptual', type=float, default=1, help='perceptual loss factor when '
                                                                                  'training generator oriented')
parser.add_argument('--image_size', type=int, default=96, help='scale images to this size. default:96')
parser.add_argument('--upscale_factor', type=int, default=4, help='scale factor for super-resolution. default:4')
parser.add_argument('--lr_img_type', type=str, default='imagenet-norm')
parser.add_argument('--hr_img_type', type=str, default='[-1, 1]')
parser.add_argument('--initial_path', type=str, default=None, help='The path of initial model for SR. defalut:None')
parser.add_argument('--teacher_initial_path', type=str, default=None, help='The path of initial teacher model for SR. defalut:None')

# noise gan
parser.add_argument('--z_dim', type=int, default=128, help='the noise input size. default:128')
parser.add_argument('--center_crop', action='store_true')



def print_options(opt, parser):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'config.txt'.format(opt.phase))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def parse():

    opt = parser.parse_args()
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)

    if opt.model == 'pix2pix' or opt.model == 'newpix2pix':
        # create model
        opt.norm = 'batch'
        opt.dataset_mode = 'aligned'
        opt.no_flip = True
        opt.load_size = 256
        opt.pool_size = 0
        opt.teacher_ndf = 128
        opt.lambda_L1 = 100.0
        if 'cityscapes' in opt.dataroot:
            opt.direction = 'BtoA'
            opt.save_epoch_freq = 5
            opt.n_epochs = 100
            opt.n_epochs_decay = 150
            opt.print_freq = 100
        if 'edges2shoes' in opt.dataroot:
            opt.batch_size = 4
            opt.n_epochs = 10
            opt.n_epochs_decay = 30
        if 'maps' in opt.dataroot:
            opt.n_epochs = 100
            opt.direction = 'BtoA'
            opt.no_flip = False
            opt.load_size = 286
            opt.n_epochs_decay = 200
            opt.save_epoch_freq = 5
            opt.print_freq = 100
            opt.lambda_L1 = 10.0
    elif opt.model == 'srgan':
        opt.dataset_mode = 'sr'
        opt.gan_mode = 'vanilla'
        opt.lr = 1e-4
        if opt.generator_only:
            opt.n_epochs = 130
            opt.n_epochs_decay = 0
            opt.batch_size = 16
        else:
            opt.n_epochs = 30
            opt.n_epochs_decay = 0
            opt.batch_size = 16
            opt.lr_policy = 'step'
            opt.lr_decay_iters = opt.n_epochs // 2
    elif opt.model == 'sagan':
        opt.dataset_mode = 'sa'
        opt.crop_size = 64
        opt.batch_size = 64
        opt.lr = 1e-4
        opt.n_epochs_decay = 0
        opt.save_epoch_freq = 5
        if 'church' in opt.dataroot:
            opt.n_epochs = 300
            opt.center_crop = False
        else:
            opt.n_epochs = 100
            opt.center_crop = True
    elif 'cyclegan' in opt.model:
        opt.dataset_mode = 'unaligned'
        opt.gan_mode = 'lsgan'
        opt.n_epochs = 100
        opt.n_epochs_decay = 100
        opt.print_freq = 100

    if opt.lambda_weight > 0 or opt.lambda_scale > 0:
        opt.n_epochs //= 10
        opt.n_epochs_decay //= 10

    # print_options(opt, parser)

    return opt