from options import options
from data import create_split_dataset
from models import get_model_class
import utils.util as util
from utils.best_information import BestInfomation
from metric.test_metric import test_pix2pix_fid, test_pix2pix_mIoU, test_srgan_psnr, test_sagan_fid, test_cyclegan_fid
from utils.prune_util import prune, cyclegan_prune

import time
import os
import copy


def test(model, opt, logger, epoch, best_performance):

    if 'pix2pix' in opt.model:

        if 'cityscapes' in opt.dataroot:
            miou = test_pix2pix_mIoU(model, copy.deepcopy(opt))
            logger.info('mIoU: %.2f' % miou)

            if best_performance.update(miou, epoch):
                model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'),
                                  fid=miou, isbest=True, direction=opt.direction)
            return miou
        else:
            fid = test_pix2pix_fid(model, copy.deepcopy(opt))
            logger.info('FID: %.2f' % fid)

            if best_performance.update(fid, epoch):
                model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'),
                          fid=fid, isbest=True, direction=opt.direction)
            return fid
    elif opt.model == 'srgan':

        test_dataset_names = ['Set5', 'Set14', 'B100', 'Urban100']

        psnrs = []
        ssims = []
        for i, dataset_name in enumerate(test_dataset_names):

            psnr, ssim = test_srgan_psnr(model, opt, dataset_name)
            psnrs.append(psnr)
            ssims.append(ssim)
            logger.info('%s:PSNR: %.2f| SSIM: %.2f' % (dataset_name, psnr, ssim))

            if best_performance.update(psnr, epoch, index=i):
                model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'),
                                  fid=psnr, isbest=True, direction=dataset_name)
            if best_performance.update(ssim, epoch, index=i + 4):
                model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'),
                                  fid=ssim, isbest=True, direction=dataset_name)
        return psnrs + ssims

    elif opt.model == 'sagan':
        fid = test_sagan_fid(model, copy.deepcopy(opt))
        logger.info('FID: %.2f' % fid)

        if best_performance.update(fid, epoch):
            model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'),
                              fid=fid, isbest=True, direction=opt.direction)
        return fid
    elif 'cyclegan' in opt.model:

        AtoB_fid, BtoA_fid = test_cyclegan_fid(model, copy.deepcopy(opt))
        logger.info('AtoB FID: %.2f | BtoA FID: %.2f' % (AtoB_fid, BtoA_fid))
        if best_performance.update(AtoB_fid, epoch, index=0):
            model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'),
                              fid=AtoB_fid, isbest=True, direction='AtoB')
        if best_performance.update(BtoA_fid, epoch, index=1):
            model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'),
                              fid=BtoA_fid, isbest=True, direction='BtoA')
        return [AtoB_fid, BtoA_fid]

if __name__ == '__main__':

    opt = options.parse()
    opt.isTrain = True
    util.mkdirs(os.path.join(opt.checkpoints_dir, opt.name))
    logger = util.get_logger(os.path.join(opt.checkpoints_dir, opt.name, 'logger.log'))

    best_performance = BestInfomation(opt)

    model_class = get_model_class(opt)
    model = model_class(opt)

    if opt.norm_prune or opt.scale_prune:
        if 'cyclegan' in opt.model:
            model = cyclegan_prune(model, opt, logger)
        else:
            model = prune(model, opt, logger)
    if opt.online_distillation:
        teacher_opt = copy.deepcopy(opt)
        teacher_opt.ngf = opt.teacher_ngf
        teacher_opt.ndf = opt.teacher_ndf
        teacher_opt.darts_discriminator = False
        teacher_opt.online_distillation = False
        teacher_opt.generator_only = False
        teacher_model = model_class(teacher_opt)
        teacher_model.model_train()
        if opt.teacher_initial_path is not None:
            teacher_model.load_models(opt.teacher_initial_path, load_discriminator=False) # teacher SR model
        setattr(model, 'teacher_model', teacher_model)
        model.init_distillation()
        teacher_model.init_distillation()

    if opt.initial_path is not None:
        model.load_models(opt.initial_path, load_discriminator=False)

    dataset_names = str(opt.dataroot).split('/')
    if str.endswith(opt.dataroot, '/'):
        dataset_name = dataset_names[-2]
    else:
        dataset_name = dataset_names[-1]

    train_dataset, val_dataset = create_split_dataset(opt)
    dataset_size = len(train_dataset)
    logger.info('The number of training images = %d' % dataset_size)

    total_iters = 0
    all_total_iters = dataset_size * opt.batch_size * (opt.n_epochs + opt.n_epochs_decay)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):

        model.model_train()
        logger.info('\nEpoch:%d' % epoch)

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        val_dataloader = iter(val_dataset)

        for i, data in enumerate(train_dataset):

            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            else:
                t_data = 0

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if opt.darts_discriminator and model.teacher_model is not None:
                val_data = next(val_dataloader)
                model.set_input(val_data)
                model.clipping_mask_alpha()
                model.optimizer_netD_arch()

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, epoch_iter, t_comp, t_data)
                for k, v in losses.items():
                    message += '%s: %.3f ' % (k, v)
                logger.info(message)
                iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:

            current_performance = test(model, copy.deepcopy(opt), logger, epoch, best_performance)
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            if epoch == opt.n_epochs + opt.n_epochs_decay:
                model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'))
        model.print_sparse_info(logger)
        logger.info('End of epoch %d / %d \t Time Taken: %d sec' %  (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        model.update_learning_rate(epoch)

    best_performance.print(logger, current_performance)
