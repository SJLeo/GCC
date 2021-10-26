class BestInfomation:

    def __init__(self, opt):

        self.opt = opt
        if 'pix2pix' in opt.model:
            self.best_metric = [0.0] if 'cityscapes' in opt.dataroot else [float('inf')]
            self.best_epoch = [0]
        elif opt.model == 'srgan':
            self.best_metric = [0.0] * 8
            self.best_epoch = [0] * 8
        elif opt.model == 'sagan':
            self.best_metric = [float('inf')] # FID
            self.best_epoch = [0]
        elif 'cyclegan' in opt.model:
            self.best_metric = [float('inf')] * 2
            self.best_epoch = [0] * 2

    def update(self, metric, epoch, index=0):

        if self.opt.model == 'srgan' or 'cityscapes' in self.opt.dataroot:
            if self.best_metric[index] <= metric:
                self.best_metric[index] = metric
                self.best_epoch[index] = epoch
                return True
        else:
            if self.best_metric[index] >= metric:
                self.best_metric[index] = metric
                self.best_epoch[index] = epoch
                return True

        return False

    def print(self, logger, last_metric):

        if 'pix2pix' in self.opt.model:
            logger.info('Best Epoch %d:%.2f/%.2f' % (self.best_epoch[0], self.best_metric[0], last_metric))
        elif self.opt.model == 'srgan':
            dataset_names = ['Set5', 'Set14', 'B100', 'Urban100']
            msg_psnr = 'Set5/Set14/B100/Urban100 PSNR: Best '
            msg_ssim = 'Set5/Set14/B100/Urban100 SSIM: Best '
            for i in range(len(dataset_names)):

                msg_psnr += 'Epoch %d:%.2f/%.2f | ' % (self.best_epoch[i],
                                                        self.best_metric[i], last_metric[i])
                msg_ssim += 'Epoch %d:%.2f/%.2f | ' % (self.best_epoch[i + 4],
                                                                     self.best_metric[i + 4], last_metric[i + 4])
            logger.info(msg_psnr + msg_ssim)
        elif self.opt.model == 'sagan':
            msg = 'FID Best Epoch %d:%.2f/%.2f' % (self.best_epoch[0], self.best_metric[0], last_metric)
            logger.info(msg)
        elif 'cyclegan' in self.opt.model:
            msg = 'FID Best AtoB Epoch %d:%.2f/%.2f | ' % (self.best_epoch[0], self.best_metric[0], last_metric[0])
            msg += 'FID Best BtoA Epoch %d:%.2f/%.2f | ' % (self.best_epoch[1], self.best_metric[1], last_metric[1])
            logger.info(msg)