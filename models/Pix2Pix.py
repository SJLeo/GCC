import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GANLoss import GANLoss
import utils.util as util

import functools
import os
from collections import OrderedDict
from models.DifferentiableOp import DifferentiableOP

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, conv_inchannel, conv_outchannel, upconv_inchannel, upconv_outchannel, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        downconv = nn.Conv2d(conv_inchannel, conv_outchannel, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(conv_outchannel)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(upconv_outchannel)
        identity = Identity()

        if outermost:
            upconv = nn.ConvTranspose2d(upconv_inchannel, upconv_outchannel,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(upconv_inchannel, upconv_outchannel,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(upconv_inchannel, upconv_outchannel,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:

                if submodule is None:
                    submodule = identity
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                if submodule is None:
                    submodule = identity
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)


    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetGenertor(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 filter_cfgs=None, channel_cfgs=None):
        super(UnetGenertor, self).__init__()

        unet_block = None

        if filter_cfgs is None or (filter_cfgs[7] != 0 and filter_cfgs[8] != 0):
            unet_block = UnetSkipConnectionBlock(conv_inchannel=ngf * 8 if channel_cfgs is None else channel_cfgs[6],
                                                 conv_outchannel=ngf * 8 if filter_cfgs is None else filter_cfgs[7],
                                                 upconv_inchannel=ngf * 8 if channel_cfgs is None else channel_cfgs[7],
                                                 upconv_outchannel=ngf * 8 if filter_cfgs is None else filter_cfgs[8],
                                                 submodule=None, norm_layer=norm_layer, innermost=True)


        for i in range(num_downs - 5):

            if filter_cfgs is None or (filter_cfgs[6 - i] != 0 and filter_cfgs[9 + i] != 0):
                unet_block = UnetSkipConnectionBlock(conv_inchannel=ngf * 8 if channel_cfgs is None else channel_cfgs[5-i],
                                                 conv_outchannel=ngf * 8 if filter_cfgs is None else filter_cfgs[6-i],
                                                 upconv_inchannel=ngf * 16 if channel_cfgs is None else channel_cfgs[8+i],
                                                 upconv_outchannel=ngf * 8 if filter_cfgs is None else filter_cfgs[9+i],
                                                 submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)


        unet_block = UnetSkipConnectionBlock(conv_inchannel=ngf * 4 if channel_cfgs is None else channel_cfgs[2],
                                             conv_outchannel=ngf * 8 if filter_cfgs is None else filter_cfgs[3],
                                             upconv_inchannel=ngf * 16 if channel_cfgs is None else channel_cfgs[11],
                                             upconv_outchannel=ngf * 4 if filter_cfgs is None else filter_cfgs[12],
                                             submodule=unet_block, norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(conv_inchannel=ngf * 2 if channel_cfgs is None else channel_cfgs[1],
                                             conv_outchannel=ngf * 4 if filter_cfgs is None else filter_cfgs[2],
                                             upconv_inchannel=ngf * 8 if channel_cfgs is None else channel_cfgs[12],
                                             upconv_outchannel=ngf * 2 if filter_cfgs is None else filter_cfgs[13],
                                             submodule=unet_block, norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(conv_inchannel=ngf if channel_cfgs is None else channel_cfgs[0],
                                             conv_outchannel=ngf * 2 if filter_cfgs is None else filter_cfgs[1],
                                             upconv_inchannel=ngf * 4 if channel_cfgs is None else channel_cfgs[13],
                                             upconv_outchannel=ngf if filter_cfgs is None else filter_cfgs[14],
                                             submodule=unet_block, norm_layer=norm_layer)

        self.model = UnetSkipConnectionBlock(conv_inchannel=input_nc,
                                conv_outchannel=ngf if filter_cfgs is None else filter_cfgs[0],
                                upconv_inchannel=ngf * 2 if channel_cfgs is None else channel_cfgs[14],
                                upconv_outchannel=output_nc,
                                submodule=unet_block, norm_layer=norm_layer, outermost=True)

    def forward(self, x):
        return self.model(x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=nn.InstanceNorm2d,
                 use_bias=True, scale_factor=1):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * scale_factor, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=in_channels, bias=use_bias),
            norm_layer(in_channels * scale_factor),
            nn.Conv2d(in_channels=in_channels * scale_factor, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=use_bias),
        )

    def forward(self, x):
        return self.conv(x)

class MobileResnetBlock(nn.Module):

    def __init__(self, layer1_input_dim, layer1_output_dim, layer2_output_dim, padding_type, norm_layer, dropout_rate,
                 use_bias, opt=None):
        super(MobileResnetBlock, self).__init__()
        self.opt = opt
        self.conv_block = self.build_conv_block(layer1_input_dim, layer1_output_dim, layer2_output_dim, padding_type,
                                                norm_layer, dropout_rate, use_bias)

    def build_conv_block(self, layer1_input_dim, layer1_output_dim, layer2_output_dim, padding_type, norm_layer,
                         dropout_rate, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SeparableConv2d(in_channels=layer1_input_dim, out_channels=layer1_output_dim,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(layer1_output_dim),
            nn.ReLU(True)
        ]
        conv_block += [nn.Dropout(dropout_rate)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SeparableConv2d(in_channels=layer1_output_dim, out_channels=layer2_output_dim,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(layer2_output_dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class MobileResnetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d,
                 dropout_rate=0, n_blocks=9, padding_type='reflect', opt=None, cfg=None):
        assert (n_blocks >= 0)
        super(MobileResnetGenerator, self).__init__()
        self.hello = 'world'
        self.opt = opt
        # self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        cfg_index = 0
        output_channel_num = ngf if cfg is None else cfg[cfg_index]
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, output_channel_num, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(output_channel_num),
                 nn.ReLU(True)]
        cfg_index += 1

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            input_channel_num = ngf * mult if cfg is None else cfg[cfg_index - 1]
            output_channel_num = ngf * mult * 2 if cfg is None else cfg[cfg_index]
            cfg_index += 1
            model += [nn.Conv2d(input_channel_num, output_channel_num, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(output_channel_num),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        for i in range(n_blocks):
            block_layer1_input_channel_num = ngf * mult if cfg is None else cfg[cfg_index - 1]
            block_layer1_output_channel_num = ngf * mult if cfg is None else cfg[cfg_index]
            cfg_index += 1
            block_layer2_output_channel_num = ngf * mult if cfg is None else cfg[cfg_index]
            cfg_index += 1
            if block_layer1_output_channel_num == 0:
                continue
            model += [MobileResnetBlock(block_layer1_input_channel_num,
                                    block_layer1_output_channel_num,
                                    block_layer2_output_channel_num,
                                    padding_type=padding_type, norm_layer=norm_layer,
                                    dropout_rate=dropout_rate, use_bias=use_bias, opt=self.opt)]

        output_channel_num = ngf
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            input_channel_num = ngf * mult if cfg is None or cfg_index == 0 else cfg[cfg_index - 1]
            output_channel_num = int(ngf * mult / 2) if cfg is None else cfg[cfg_index]
            cfg_index += 1
            model += [nn.ConvTranspose2d(input_channel_num, output_channel_num,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(output_channel_num, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class NLayerDiscriminator(nn.Module):
    '''Defines a PatchGAN discriminator'''

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=1, padding=padding,bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):

        return self.model(x)

class MaskNLayerDiscriminator(nn.Module):
    '''Defines a PatchGAN discriminator'''

    def __init__(self, input_nc=3, ndf=64, n_layers=3, threshold=0.5):
        super(MaskNLayerDiscriminator, self).__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding),
                    nn.LeakyReLU(0.2, True),
                    DifferentiableOP(ndf, threshold)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_layer(ndf * nf_mult),
                DifferentiableOP(ndf * nf_mult, threshold),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=1, padding=padding,bias=use_bias),
            norm_layer(ndf * nf_mult),
            DifferentiableOP(ndf * nf_mult, threshold),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):

        return self.model(x)

class Pix2PixModel(nn.Module):

    def __init__(self, opt, filter_cfgs=None, channel_cfgs=None):
        super(Pix2PixModel, self).__init__()

        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        self.filter_cfgs = filter_cfgs
        self.channel_cfgs = channel_cfgs
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.current_D_arch_diff_loss = 0.0

        if self.opt.backbone == 'resnet':
            self.generator_extract_layers = ['model.9', 'model.12', 'model.15', 'model.18']
        else:
            self.generator_extract_layers = ['model.model.1.model.2',
                                             'model.model.1.model.3.model.3.model.2',
                                             'model.model.1.model.3.model.3.model.4',
                                             'model.model.1.model.4']
        if self.opt.darts_discriminator:
            self.discriminator_extract_layers = ['model.4', 'model.12']
        else:
            self.discriminator_extract_layers = ['model.3', 'model.9']

        self.optimizers = []
        if self.opt.backbone == 'resnet':
            self.netG = MobileResnetGenerator(input_nc=3, output_nc=3, ngf=self.opt.ngf, cfg=filter_cfgs, opt=opt)
        else:
            self.netG = UnetGenertor(input_nc=3, output_nc=3, num_downs=self.opt.num_downs, ngf=opt.ngf,
                                     use_dropout=not opt.no_dropout,
                                     filter_cfgs=filter_cfgs, channel_cfgs=channel_cfgs)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        if self.opt.online_distillation or self.opt.normal_distillation:

            self.transform_convs = []
            if self.opt.backbone == 'resnet':
                teacher_extract_layer_ngf = [self.opt.teacher_ngf * 4] * 4
                if self.filter_cfgs is None:
                    student_extract_layer_ngf = [self.opt.ngf * 4] * 4
                else:
                    student_extract_layer_ngf = [self.filter_cfgs[2]] * 4
            else:
                teacher_extract_layer_ngf = [self.opt.teacher_ngf * 2, self.opt.teacher_ngf * 8,
                                             self.opt.teacher_ngf * 16, self.opt.teacher_ngf * 4]
                if self.channel_cfgs is None:
                    student_extract_layer_ngf = [self.opt.ngf * 2, self.opt.ngf * 8,
                                                 self.opt.ngf * 16, self.opt.ngf * 4]
                else:
                    student_extract_layer_ngf = [self.channel_cfgs[1], self.channel_cfgs[3],
                                                 self.channel_cfgs[-4], self.channel_cfgs[-2]]
            index = 0
            netG_parms = []
            for name, module in self.netG.named_modules():

                if name in self.generator_extract_layers:
                    self.transform_convs.append(nn.Conv2d(in_channels=student_extract_layer_ngf[index],
                                                          out_channels=teacher_extract_layer_ngf[index],
                                                          kernel_size=1, stride=1, padding=0, bias=False).to(self.device))
                    netG_parms += list(self.transform_convs[-1].parameters())
                    index += 1
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or \
                        isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.ConvTranspose2d):
                    netG_parms += list(module.parameters())
            self.optimizer_G = torch.optim.Adam(netG_parms, lr=opt.lr, betas=(0.5, 0.999))

        if self.opt.darts_discriminator:
            self.loss_names.append('D_arch_diff')
            self.loss_names.append('D_arch')
            self.loss_names.append('teacher_D_arch_diff')
            self.netD = MaskNLayerDiscriminator(input_nc=3+3, ndf=opt.ndf, threshold=self.opt.threshold)
            weight_params = []
            arch_params = []
            for name, module in self.netD.named_modules():

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d):
                    weight_params += list(module.parameters())
                if isinstance(module, DifferentiableOP):
                    arch_params += list(module.parameters())
            self.optimizer_D = torch.optim.Adam(weight_params, lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_arch = torch.optim.Adam(arch_params, lr=opt.arch_lr)
            if self.opt.arch_lr_step:
                import copy
                arch_opt = copy.deepcopy(opt)
                arch_opt.lr_policy = 'step'
                arch_opt.lr_decay_iters = opt.n_epochs - 1
                self.arch_scheduler = util.get_scheduler(self.optimizer_arch, arch_opt)
        else:
            self.netD = NLayerDiscriminator(input_nc=3+3, ndf=opt.ndf)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.init_net()

        self.criterionGAN = GANLoss(self.opt.gan_mode).to(self.device)
        self.criterionL1 = nn.L1Loss().to(self.device)
        self.criterionMSE = nn.MSELoss().to(self.device)

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.schedulers = [util.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if self.opt.darts_discriminator and self.opt.arch_lr_step:
            self.schedulers.append(self.arch_scheduler)

    def set_input(self, input):
        self.input = input
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = [input['A_paths' if AtoB else 'B_paths'], input['B_paths' if AtoB else 'A_paths']]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_D_arch(self):

        self.loss_teacher_D_arch_diff, loss_teacher_D_arch_diff_sign = self.teacher_model.get_D_arch_diff(isTeacher=True)

        self.loss_D_arch_diff, loss_D_arch_diff_sign = self.get_D_arch_diff(isTeacher=False)
        self.loss_D_arch = self.criterionL1(self.loss_D_arch_diff, self.loss_teacher_D_arch_diff)

        self.loss_D_arch += (self.loss_D_arch_real + self.loss_D_arch_fake) * 0.5
        self.loss_D_arch.backward()

    def get_D_arch_diff(self, isTeacher=False):

        if isTeacher:
            self.set_requires_grad(self.netD, False)  # teacher D requires no gradients when get loss_D_arch_diff

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_arch_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        self.loss_D_arch_fake_real = self.criterionGAN(pred_fake, True, for_discriminator=False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_arch_real = self.criterionGAN(pred_real, True, for_discriminator=True)

        if isTeacher:
            if self.current_D_arch_diff_loss != 0.0:
                self.current_D_arch_diff_loss = (self.opt.ema_beta * self.criterionL1(self.loss_D_arch_fake_real, self.loss_D_arch_fake) +
                        (1.0 - self.opt.ema_beta) * self.current_D_arch_diff_loss)
            else:
                self.current_D_arch_diff_loss = self.criterionL1(self.loss_D_arch_fake_real, self.loss_D_arch_fake)
        else:
            self.current_D_arch_diff_loss = self.criterionL1(self.loss_D_arch_fake_real, self.loss_D_arch_fake)
        return self.current_D_arch_diff_loss, torch.sign(self.loss_D_arch_fake_real - self.loss_D_arch_fake)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True, for_discriminator=False)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # combine loss and calculate gradients
        # if self.opt.online_distillation:
        #     self.loss_G = self.loss_G_L1
        # else:
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        if self.opt.online_distillation or self.opt.normal_distillation:
            self.Tfake_B = self.teacher_model.fake_B
            current_distillation_features = self.get_distillation_features()
            self.teacher_model.netD(torch.cat((self.real_A, self.fake_B), 1)) # output teacher discriminator feature for Gs
            teacher_discriminator_features = list(self.teacher_model.total_discriminator_features.values())
            current_distillation_features[-len(self.discriminator_extract_layers):] = teacher_discriminator_features[:]

            self.loss_content = 0.0
            self.loss_gram = 0.0
            for i, feature in enumerate(current_distillation_features):

                if i < 4: # generator feature should transform channel dimension consistent with teacher
                    feature = self.transform_convs[i](feature)
                target_feature = self.target_distillation_features[i]
                self.loss_gram += torch.sqrt(self.criterionMSE(self.gram(feature), self.gram(target_feature.detach())))
                self.loss_content += torch.sqrt(self.criterionMSE(feature, target_feature.detach()))

            self.loss_gram = self.opt.lambda_gram * self.loss_gram
            self.loss_content = self.opt.lambda_content * self.loss_content
            self.loss_G += self.loss_gram
            self.loss_G += self.loss_content

        self.loss_G.backward()

        self.L1_sparsity()

    def L1_sparsity(self):

        if self.opt.lambda_weight > 0.0:
            for module in self.netG.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                    module.weight.grad.data.add_(self.opt.lambda_weight * torch.sign(module.weight.data))
        elif self.opt.lambda_scale > 0.0:
            for module in self.netG.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.weight.grad.data.add_(self.opt.lambda_scale * torch.sign(module.weight.data))

    def optimize_parameters(self):

        if self.opt.online_distillation:
            self.teacher_model.set_input(self.input)
            self.teacher_model.optimize_parameters()
            self.target_distillation_features = [f.clone() for f in self.teacher_model.get_distillation_features()]

        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_netD_arch_grad(False) # disable backprop for differentiableOP
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def optimizer_netD_arch(self):
        self.forward()
        self.teacher_model.set_input(self.input)
        self.teacher_model.forward()
        self.set_requires_grad(self.netD, True)
        self.set_netD_weight_grad(False)
        self.optimizer_arch.zero_grad()
        self.backward_D_arch()
        self.optimizer_arch.step()

    def print_sparse_info(self, logger):

        for name, module in self.named_modules():
            if isinstance(module, DifferentiableOP):
                mask = module.get_current_mask()
                logger.info('%s sparsity ratio: %.2f' % (name, float(sum(mask == 0.0)) / mask.numel()))

    def adaptive_ema_beta(self, epoch):

        self.opt.ema_beta = 1.0 - epoch / (self.opt.n_epochs + self.opt.n_epochs_decay)

    def update_learning_rate(self, epoch):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()

        self.adaptive_ema_beta(epoch)

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f\tema beta = %.7f' % (lr, self.opt.ema_beta))

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_netD_weight_grad(self, requires_grad=False):
        for module in self.netD.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d):
                for param in module.parameters():
                    param.requires_grad = requires_grad

    def set_netD_arch_grad(self, requires_grad=False):
        for module in self.netD.modules():
            if isinstance(module, DifferentiableOP):
                for param in module.parameters():
                    param.requires_grad = requires_grad

    def save_models(self, epoch, save_dir, fid=None, isbest=False, direction='AtoB'):
        util.mkdirs(save_dir)
        ckpt = {
            'G': self.netG.state_dict(),
            'D': self.netD.state_dict(),
            'epoch': epoch,
            'cfg': (self.filter_cfgs, self.channel_cfgs),
            'fid': fid
        }
        if isbest:
            save_path = os.path.join(save_dir, 'model_best_%s.pth' % direction)
        else:
            save_path = os.path.join(save_dir, 'model_%d.pth' % epoch)
        torch.save(ckpt, save_path)

    def load_models(self, load_path, load_discriminator=True):
        ckpt = torch.load(load_path, map_location=self.device)
        self.netG.load_state_dict(ckpt['G'])
        if load_discriminator:
            self.netD.load_state_dict(ckpt['D'])

        print('loading the model from %s' % (load_path))
        return ckpt['fid'], float('inf')

    def init_net(self):
        self.netG.to(self.device)
        self.netD.to(self.device)

        if self.opt.darts_discriminator:

            for module in self.netD.modules():
                if isinstance(module, DifferentiableOP):
                    module.threshold = module.threshold.to(module.alpha.device)

        util.init_weights(self.netG, init_type='normal', init_gain=0.02)
        util.init_weights(self.netD, init_type='normal', init_gain=0.02)

    def clipping_mask_alpha(self):
        for module in self.netD.modules():
            if isinstance(module, DifferentiableOP):
                module.clip_alpha()

    def model_train(self):
        self.netG.train()
        self.netD.train()

    def model_eval(self):
        self.netG.eval()
        self.netD.eval()

    def get_current_visuals(self):
        """Return visualization images. """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def init_distillation(self):

        self.total_generator_features = {}
        self.total_discriminator_features = {}

        if self.opt.online_distillation or self.opt.normal_distillation:
            if self.opt.lambda_content > 0.0:
                self.loss_names.append('content')
            if self.opt.lambda_gram > 0.0:
                self.loss_names.append('gram')
            self.visual_names.append('Tfake_B')
            # if self.opt.sign:
            #     self.loss_names.append('regular')

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name] = output
            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self.netG, self.total_generator_features, self.generator_extract_layers)
        add_hook(self.netD, self.total_discriminator_features, self.discriminator_extract_layers)

    def get_distillation_features(self):

        return list(self.total_generator_features.values()) + list(self.total_discriminator_features.values())

    def gram(self, x):

        b, c, h, w = x.size()
        f = x.view(b, c, h * w)
        f_T = f.transpose(1, 2)

        G = f.bmm(f_T) / (c * h * w)
        return G

    def prune(self, threshold):

        if self.opt.backbone == 'resnet':
            return self.resnet_prune(threshold)
        else:
            if self.opt.scale_prune:
                return self.scale_prune(threshold)
            elif self.opt.norm_prune:
                return self.norm_prune(threshold)
            else:
                raise NotImplementedError('only scale and norm pruning are supported!!!')

    def max_min_bn_scale(self):

        prunable_layers = ['model.model.1.model.3.model.3.model.3.model.3.model.2',
                           'model.model.1.model.3.model.3.model.3.model.3.model.3.model.2',
                           'model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.4',
                           'model.model.1.model.3.model.3.model.3.model.3.model.3.model.6',
                           'model.model.1.model.3.model.3.model.3.model.3.model.6']

        unprunable_layers_maxscale = float('inf')
        prunable_layers_maxscale = -float('inf')
        overall_minscale = float('inf')

        for name, module in self.netG.named_modules():

            if isinstance(module, nn.BatchNorm2d):

                if name in prunable_layers:
                    prunable_layers_maxscale = max(torch.max(module.weight.data), prunable_layers_maxscale)
                else:
                    unprunable_layers_maxscale = min(torch.max(module.weight.data), unprunable_layers_maxscale)
                overall_minscale = min(torch.min(module.weight.data), overall_minscale)

        return min(prunable_layers_maxscale, unprunable_layers_maxscale), overall_minscale

    def max_min_conv_norm(self):

        overall_minnorm = float('inf')
        unprunable_layers_maxnorm = float('inf')
        prunable_layer_maxnorm = -float('inf')

        prunable_layers = [
            'model.model.1.model.3.model.3.model.3.model.3.model.1',
            'model.model.1.model.3.model.3.model.3.model.3.model.3.model.1',
            'model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.1',
            'model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3',
            'model.model.1.model.3.model.3.model.3.model.3.model.3.model.5',
            'model.model.1.model.3.model.3.model.3.model.3.model.5'
        ]

        unprunable_layer_names = ['model.26']
        for i in range(10, 19, 1):
            unprunable_layer_names.append('model.%d.conv_block.1.conv.0' % i)
            unprunable_layer_names.append('model.%d.conv_block.6.conv.0' % i)

        for name, module in self.netG.named_modules():

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):

                if name in unprunable_layer_names:
                    continue

                if isinstance(module, nn.Conv2d):
                    weight_norm = torch.sum(torch.abs(module.weight.data), (1, 2, 3))
                else:
                    weight_norm = torch.sum(torch.abs(module.weight.data), (0, 2, 3))

                if name in prunable_layers:
                    prunable_layer_maxnorm = max(torch.max(weight_norm), prunable_layer_maxnorm)
                else:
                    unprunable_layers_maxnorm = min(torch.max(weight_norm), unprunable_layers_maxnorm)
                overall_minnorm = min(torch.min(weight_norm), overall_minnorm)
        if self.opt.backbone == 'resnet':
            return unprunable_layers_maxnorm, overall_minnorm
        else:
            return min(prunable_layer_maxnorm, unprunable_layers_maxnorm), overall_minnorm

    def get_cfg(self):
        return self.filter_cfgs, self.channel_cfgs

    def scale_prune(self, threshold):

        filter_cfgs = [self.opt.ngf]
        channel_cfgs = [self.opt.ngf]

        upconv_flag = False
        upconv_num = 0
        for name, module in self.netG.named_modules():

            if isinstance(module, nn.BatchNorm2d):

                bn_weight = module.weight.data
                mask = bn_weight > threshold
                filter_cfg = int(sum(mask))
                filter_cfgs.append(filter_cfg)

                if name == 'model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.4':
                    upconv_flag = True
                    if filter_cfg == 0:
                        filter_cfgs[-2] = 0

                if upconv_flag:
                    upconv_num += 1
                    if filter_cfgs[(-2) * upconv_num] == 0:
                        filter_cfgs[-1] = 0
                        filter_cfg = 0
                    channel_cfgs.append(filter_cfg + filter_cfgs[-1 + (-2) * upconv_num])
                else:
                    channel_cfgs.append(filter_cfg)

                if name == 'model.model.1.model.3.model.3.model.3.model.3.model.3.model.2':

                    if filter_cfgs[-1] == 0:
                        filter_cfgs.append(0)
                        channel_cfgs.append(0)
                    else:
                        filter_cfgs.append(self.opt.ngf * 8)
                        channel_cfgs.append(self.opt.ngf * 8)

        pruned_model = Pix2PixModel(self.opt, filter_cfgs=filter_cfgs, channel_cfgs=channel_cfgs)

        return pruned_model

    def norm_prune(self, threshold):

        filter_cfgs = []
        channel_cfgs = []

        upconv_num = 0
        for name, module in self.netG.named_modules():

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):

                weight = module.weight.data
                if isinstance( module, nn.Conv2d):
                    weight_norm = torch.sum(torch.abs(weight), (1, 2, 3))
                else:
                    weight_norm = torch.sum(torch.abs(weight), (0, 2, 3))
                mask = weight_norm > threshold
                filter_cfg = int(sum(mask))
                filter_cfgs.append(filter_cfg)

                if isinstance(module, nn.ConvTranspose2d):
                    upconv_num += 1
                    if name != 'model.model.3':
                        channel_cfgs.append(filter_cfg + filter_cfgs[-1 + (-2) * upconv_num])
                else:
                    channel_cfgs.append(filter_cfg)

        # print(filter_cfgs)
        # print(channel_cfgs)
        # print('-------------------')
        if filter_cfgs[0] == 0:
            filter_cfgs[0] = self.opt.ngf
            channel_cfgs[0] = self.opt.ngf
            channel_cfgs[-1] += self.opt.ngf

        pruned_model = Pix2PixModel(self.opt, filter_cfgs=filter_cfgs, channel_cfgs=channel_cfgs)

        return pruned_model

    def resnet_prune(self, threshold):

        filter_cfgs = []

        unprunable_layer_names = ['model.26']
        for i in range(10, 19, 1):
            unprunable_layer_names.append('model.%d.conv_block.1.conv.0' % i)
            unprunable_layer_names.append('model.%d.conv_block.6.conv.0' % i)

        residual_conv_names = ['model.7']
        for i in range(10, 19, 1):
            residual_conv_names.append('model.%d.conv_block.6.conv.2' % i)
        state_dict = self.netG.state_dict()
        residual_width = state_dict[residual_conv_names[0] + '.weight'].size(0)
        residual_mask = [0] * residual_width
        for name in residual_conv_names:
            weight_norm = torch.sum(torch.abs(state_dict[name + '.weight']), (1, 2, 3))
            current_mask = weight_norm > threshold
            for i, mask in enumerate(current_mask):
                if mask:
                    residual_mask[i] += 1
        residual_mask = torch.FloatTensor(residual_mask) > 0

        for name, module in self.netG.named_modules():

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):

                if name in unprunable_layer_names:
                    continue

                if name in residual_conv_names:
                    filter_cfgs.append(int(sum(residual_mask)))
                else:
                    weight = module.weight.data
                    if isinstance( module, nn.Conv2d):
                        weight_norm = torch.sum(torch.abs(weight), (1, 2, 3))
                    else:
                        weight_norm = torch.sum(torch.abs(weight), (0, 2, 3))
                    mask = weight_norm > threshold
                    filter_cfg = int(sum(mask))
                    filter_cfgs.append(filter_cfg)

        # print(filter_cfgs)
        # print(channel_cfgs)
        # print('-------------------')

        pruned_model = Pix2PixModel(self.opt, filter_cfgs=filter_cfgs)

        return pruned_model
