from models import Pix2Pix, SRGAN, SAGAN, CycleGAN

def get_model_class(opt):
    if opt.model == 'pix2pix':
        # create model
        model_class = Pix2Pix.Pix2PixModel
    elif opt.model == 'srgan':
        model_class = SRGAN.SRGAN
    elif opt.model == 'sagan':
        model_class = SAGAN.SAGANModel
    elif opt.model == 'cyclegan':
        model_class = CycleGAN.MobileCycleGANModel
    else:
        raise NotImplementedError('%s not implemented' % opt.model)
    return model_class