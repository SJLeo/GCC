import torch
from thop import profile
from models import get_model_class
# from models.CycleGAN import MobileCycleGANModel

def get_flops_parms(model, device, opt, verbose=False):

    if 'sr' in opt.dataroot:
        input = torch.randn(1, 3, opt.image_size // (opt.upscale_factor),
                            opt.image_size // (opt.upscale_factor)).to(device)
    elif 'celeb' in opt.dataroot or 'church' in opt.dataroot:
        input = torch.randn(1, opt.z_dim).to(device)
    else:
        input = torch.randn(1, 3, opt.load_size, opt.load_size).to(device)

    macs, params = profile(model, inputs=(input,), verbose=verbose)

    return macs / (1000 ** 3), params / (1000 ** 2) # macs unit is G, params unit is M

def binarysearch_threshold(model, target_budget):

    if model.opt.scale_prune:
        max_scale, min_scale = model.max_min_bn_scale()
    else:
        max_scale, min_scale = model.max_min_conv_norm()
    # print(max_scale, min_scale)

    if 'sr' in model.opt.dataroot:
        tolerance = 0.01
    elif 'celeb' in model.opt.dataroot or 'church' in model.opt.dataroot:
        tolerance = 0.001
    else:
        tolerance = 0.1

    while max_scale > min_scale:

        mid_scale = (max_scale + min_scale) / 2
        pruned_model = model.prune(mid_scale)
        budget, _ = get_flops_parms(pruned_model.netG, pruned_model.device, pruned_model.opt)
        if abs(target_budget - budget) <= tolerance:
            return mid_scale
        elif target_budget - budget > tolerance:
            max_scale = mid_scale
        else:
            min_scale = mid_scale

    raise NotImplementedError('No appropriate threshold found')

def prune(model, opt, logger):
    if opt.target_budget is None:
        raise NotImplementedError('the target budget must be exist!!!')
    if opt.pretrain_path is None:
        raise NotImplementedError('the pretrain path must be exist!!!')
    model.load_models(opt.pretrain_path, load_discriminator=False)

    threshold = binarysearch_threshold(model, opt.target_budget)
    pruned_model = model.prune(threshold, lottery_path=opt.lottery_path)
    filter_cfg, channel_cfg = pruned_model.get_cfg()
    macs, params = get_flops_parms(pruned_model.netG, pruned_model.device, pruned_model.opt)
    logger.info(filter_cfg)
    logger.info(channel_cfg)
    logger.info('MACs:%.7f G  |  Params:%.4f M' % (macs, params))
    return pruned_model

def cyclegan_binarysearch_cfg(model, target_budget, target_budget_B):

    max_scale_A, min_scale_A = model.max_min_conv_norm(model.netG_A)
    max_scale_B, min_scale_B = model.max_min_conv_norm(model.netG_B)

    tolerance = 0.05
    final_cfg_AtoB = None
    final_cfg_BtoA = None
    model_class = get_model_class(model.opt)

    while max_scale_A > min_scale_A:

        mid_scale = (max_scale_A + min_scale_A) / 2
        cfg_AtoB = model.get_prunenet_cfg(model.netG_A, mid_scale)
        pruned_model = model_class(model.opt, cfg_AtoB=cfg_AtoB)
        budget, _ = get_flops_parms(pruned_model.netG_A, pruned_model.device, pruned_model.opt)
        print(mid_scale, budget)
        if abs(target_budget - budget) <= tolerance:
            final_cfg_AtoB = cfg_AtoB
            break
        elif target_budget - budget > tolerance:
            max_scale_A = mid_scale
        else:
            min_scale_A = mid_scale
    print('--------------------')
    while max_scale_B > min_scale_B:

        mid_scale = (max_scale_B + min_scale_B) / 2
        cfg_BtoA = model.get_prunenet_cfg(model.netG_B, mid_scale)
        pruned_model = model_class(model.opt, cfg_BtoA=cfg_BtoA)
        budget, _ = get_flops_parms(pruned_model.netG_B, pruned_model.device, pruned_model.opt)
        print(mid_scale, budget)
        if abs(target_budget_B - budget) <= tolerance:
            final_cfg_BtoA = cfg_BtoA
            break
        elif target_budget_B - budget > tolerance:
            max_scale_B = mid_scale
        else:
            min_scale_B = mid_scale
    print(final_cfg_AtoB, final_cfg_BtoA)
    if final_cfg_AtoB is None or final_cfg_BtoA is None:
        raise NotImplementedError('No appropriate threshold found')
    else:
        return final_cfg_AtoB, final_cfg_BtoA

def cyclegan_prune(model, opt, logger):
    if opt.target_budget is None or opt.target_budget_B is None:
        raise NotImplementedError('the target budget must be exist!!!')
    if opt.pretrain_path is None:
        raise NotImplementedError('the pretrain path must be exist!!!')
    model.load_models(opt.pretrain_path, load_discriminator=False)
    model_class = get_model_class(model.opt)

    cfg_AtoB, cfg_BtoA = cyclegan_binarysearch_cfg(model, opt.target_budget, opt.target_budget_B)
    # cfg_AtoB = [19, 29, 104, 128, 104, 123, 104, 111, 104, 98, 104, 84, 104, 82, 104, 74, 104, 81, 104, 87, 104, 28, 29]
    cfg_AtoB = [24, 48, 86, 72, 86, 47, 86, 44, 86, 43, 86, 43, 86, 29, 86, 30, 86, 37, 86, 36, 86, 48, 24]
    cfg_BtoA = [24, 48, 96, 91, 96, 73, 96, 62, 96, 61, 96, 74, 96, 54, 96, 51, 96, 58, 96, 81, 96, 48, 24]
    pruned_model = model_class(model.opt, cfg_AtoB=cfg_AtoB, cfg_BtoA=cfg_BtoA)
    cfg_AtoB, cfg_BtoA = pruned_model.get_cfg()
    logger.info(cfg_AtoB)
    logger.info(cfg_BtoA)
    macs, params = get_flops_parms(pruned_model.netG_A, pruned_model.device, pruned_model.opt)
    logger.info('netG_A MACs:%.7f G  |  Params:%.4f M' % (macs, params))
    macs, params = get_flops_parms(pruned_model.netG_B, pruned_model.device, pruned_model.opt)
    logger.info('netG_B MACs:%.7f G  |  Params:%.4f M' % (macs, params))
    return pruned_model