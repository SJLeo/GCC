import torch
import torch.nn as nn

import utils.util as util
from utils.image_pool import ImagePool
from models.GANLoss import GANLoss
from models.DifferentiableOp import DifferentiableOP

import itertools
import functools
import os
import copy
from collections import OrderedDict

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
    def __init__(self, layer1_input_dim, layer1_output_dim, layer2_output_dim, padding_type, norm_layer, dropout_rate, use_bias):
        super(MobileResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(layer1_input_dim, layer1_output_dim, layer2_output_dim, padding_type, norm_layer, dropout_rate, use_bias)

    def build_conv_block(self, layer1_input_dim, layer1_output_dim, layer2_output_dim, padding_type, norm_layer, dropout_rate, use_bias):
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
                 dropout_rate=0, n_blocks=9, padding_type='reflect', cfg=None):
        assert (n_blocks >= 0)
        super(MobileResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        cfg_index = 0
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf if cfg is None else cfg[cfg_index], kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
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
                                    dropout_rate=dropout_rate, use_bias=use_bias)]

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
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
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
                    DifferentiableOP(ndf, threshold=threshold)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_layer(ndf * nf_mult),
                DifferentiableOP(ndf * nf_mult, threshold=threshold),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=1, padding=padding,bias=use_bias),
            norm_layer(ndf * nf_mult),
            DifferentiableOP(ndf * nf_mult, threshold=threshold),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):

        return self.model(x)

class MobileCycleGANModel(nn.Module):

    def __init__(self, opt, cfg_AtoB=None, cfg_BtoA=None):
        super(MobileCycleGANModel, self).__init__()
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        self.cfg_AtoB = cfg_AtoB
        self.cfg_BtoA = cfg_BtoA
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'idt_B']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'idt_A']
        self.visual_names = visual_names_A + visual_names_B
        self.current_netD_A_arch_diff_loss = 0.0
        self.current_netD_B_arch_diff_loss = 0.0

        # self.generator_extract_layers = ['model.6', 'model.11', 'model.16', 'model.21']
        self.generator_extract_layers = ['model.9', 'model.12', 'model.15', 'model.18']

        if self.opt.darts_discriminator:
            self.discriminator_extract_layers = ['model.4', 'model.12']
        else:
            self.discriminator_extract_layers = ['model.3', 'model.9']
        self.heavy_sparsity = ['model.1', 'model.4', 'model.19', 'model.22']


        self.netG_A = MobileResnetGenerator(ngf=self.opt.ngf, cfg=cfg_AtoB)
        self.netG_B = MobileResnetGenerator(ngf=self.opt.ngf, cfg=cfg_BtoA)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=opt.lr, betas=(0.5, 0.999))

        if self.opt.online_distillation or self.opt.normal_distillation:
            teacher_extract_layer_ngf = [self.opt.teacher_ngf * 4] * 4
            # teacher_extract_layer_ngf = [self.opt.teacher_ngf * 2, self.opt.teacher_ngf * 4, self.opt.teacher_ngf * 4, self.opt.teacher_ngf * 2]
            if self.cfg_AtoB is None:
                studentA_extract_layer_ngf = [self.opt.ngf * 4] * 4
            else:
                studentA_extract_layer_ngf = [self.cfg_AtoB[2]] * 4
                # studentA_extract_layer_ngf = [self.cfg_AtoB[1], self.cfg_AtoB[2], self.cfg_AtoB[2], self.cfg_AtoB[-2]]
            if self.cfg_BtoA is None:
                studentB_extract_layer_ngf = [self.opt.ngf * 4] * 4

            else:
                studentB_extract_layer_ngf = [self.cfg_BtoA[2]] * 4
                # studentB_extract_layer_ngf = [self.cfg_BtoA[1], self.cfg_BtoA[2], self.cfg_BtoA[2], self.cfg_BtoA[-2]]
            index = 0
            netG_A_params = []
            self.transform_A_convs = []
            for name, module in self.netG_A.named_modules():

                if name in self.generator_extract_layers:
                    self.transform_A_convs.append(nn.Conv2d(in_channels=studentA_extract_layer_ngf[index],
                                                          out_channels=teacher_extract_layer_ngf[index],
                                                          kernel_size=1, stride=1, padding=0, bias=False).to(self.device))
                    netG_A_params += list(self.transform_A_convs[-1].parameters())
                    index += 1
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or \
                        isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.ConvTranspose2d):
                    netG_A_params += list(module.parameters())

            index = 0
            netG_B_params = []
            self.transform_B_convs = []
            for name, module in self.netG_B.named_modules():

                if name in self.generator_extract_layers:
                    self.transform_B_convs.append(nn.Conv2d(in_channels=studentB_extract_layer_ngf[index],
                                                            out_channels=teacher_extract_layer_ngf[index],
                                                            kernel_size=1, stride=1, padding=0, bias=False).to(
                        self.device))
                    netG_B_params += list(self.transform_B_convs[-1].parameters())
                    index += 1
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or \
                        isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.ConvTranspose2d):
                    netG_B_params += list(module.parameters())
            self.optimizer_G = torch.optim.Adam(itertools.chain(netG_A_params, netG_B_params),
                                                lr=opt.lr, betas=(0.5, 0.999))

        if self.opt.darts_discriminator:
            self.loss_names.append('D_arch_diff_A')
            self.loss_names.append('D_arch_A')
            self.loss_names.append('D_arch_diff_B')
            self.loss_names.append('D_arch_B')
            self.loss_names.append('teacher_netD_A_arch_diff')
            self.loss_names.append('teacher_netD_B_arch_diff')
            self.netD_A = MaskNLayerDiscriminator(ndf=self.opt.ndf, threshold=self.opt.threshold)
            self.netD_B = MaskNLayerDiscriminator(ndf=self.opt.ndf, threshold=self.opt.threshold)
            netD_A_weight_params = []
            netD_A_arch_params = []
            for name, module in self.netD_A.named_modules():

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d):
                    netD_A_weight_params += list(module.parameters())
                if isinstance(module, DifferentiableOP):
                    netD_A_arch_params += list(module.parameters())
            netD_B_weight_params = []
            netD_B_arch_params = []
            for name, module in self.netD_B.named_modules():

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module,
                                                                                                     nn.InstanceNorm2d):
                    netD_B_weight_params += list(module.parameters())
                if isinstance(module, DifferentiableOP):
                    netD_B_arch_params += list(module.parameters())
            self.optimizer_D = torch.optim.Adam(itertools.chain(netD_A_weight_params, netD_B_weight_params),
                                                lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_arch = torch.optim.Adam(itertools.chain(netD_A_arch_params, netD_B_arch_params),
                                                lr=opt.arch_lr)
            if self.opt.arch_lr_step:
                arch_opt = copy.deepcopy(opt)
                arch_opt.lr_policy = 'step'
                arch_opt.lr_decay_iters = opt.n_epochs - 1
                self.arch_scheduler = util.get_scheduler(self.optimizer_arch, arch_opt)
        else:
            self.netD_A = NLayerDiscriminator(ndf=self.opt.ndf)
            self.netD_B = NLayerDiscriminator(ndf=self.opt.ndf)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(0.5, 0.999))
        self.init_net()

        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        self.teacher_model = None

        # define loss functions
        self.criterionGAN= GANLoss(opt.gan_mode).to(self.device)
        self.criterionCycle = nn.L1Loss().to(self.device)
        self.criterionIdt = nn.L1Loss().to(self.device)
        self.criterionMSE = nn.MSELoss().to(self.device)
        self.criterionL1 = nn.L1Loss().to(self.device)


        # define optimizers
        self.optimizers = []
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

        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        self.idt_A = self.netG_A(self.real_B)
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        self.idt_B = self.netG_B(self.real_A)

        self.fake_B = self.netG_A(self.real_A)  # G_A(A) second to get hook for distillation
        self.fake_A = self.netG_B(self.real_B)  # G_B(B) second to get hook for distillation

    def visual_forward(self):

        self.fake_B = self.netG_A(self.real_A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_D_arch(self):
        self.loss_teacher_netD_A_arch_diff, self.loss_teacher_netD_B_arch_diff = self.teacher_model.get_D_arch_diff(isTeacher=True)
        self.loss_D_arch_diff_A, self.loss_D_arch_diff_B = self.get_D_arch_diff(isTeacher=False)
        self.loss_D_arch_A = self.criterionL1(self.loss_D_arch_diff_A, self.loss_teacher_netD_A_arch_diff)
        self.loss_D_arch_B = self.criterionL1(self.loss_D_arch_diff_B, self.loss_teacher_netD_B_arch_diff)

        self.loss_D_arch_A += (self.loss_D_A_arch_fake + self.loss_D_A_arch_real) * 0.5
        self.loss_D_arch_B += (self.loss_D_B_arch_fake + self.loss_D_B_arch_real) * 0.5
        self.loss_D_arch_A.backward()
        self.loss_D_arch_B.backward()

    def get_D_arch_diff(self, isTeacher=False):

        if isTeacher:
            self.set_requires_grad([self.netD_A, self.netD_B], False) # teacher D requires no gradients when get loss_D_arch_diff

        pred_fake_netA = self.netD_A(self.fake_B.detach())
        self.loss_D_A_arch_fake = self.criterionGAN(pred_fake_netA, False, for_discriminator=True)
        self.loss_D_A_arch_fake_real = self.criterionGAN(pred_fake_netA, True, for_discriminator=False)

        pred_real_A = self.netD_A(self.real_B)
        self.loss_D_A_arch_real = self.criterionGAN(pred_real_A, True, for_discriminator=True)

        pred_fake_netB = self.netD_B(self.fake_A.detach())
        self.loss_D_B_arch_fake = self.criterionGAN(pred_fake_netB, False, for_discriminator=True)
        self.loss_D_B_arch_fake_real = self.criterionGAN(pred_fake_netB, True, for_discriminator=False)

        pred_real_B = self.netD_B(self.real_A)
        self.loss_D_B_arch_real = self.criterionGAN(pred_real_B, True, for_discriminator=True)

        if isTeacher:
            if self.current_netD_A_arch_diff_loss != 0.0:
                self.current_netD_A_arch_diff_loss = \
                    (self.opt.ema_beta * self.criterionL1(self.loss_D_A_arch_fake_real, self.loss_D_A_arch_fake) +
                     (1.0 - self.opt.ema_beta) * self.current_netD_A_arch_diff_loss)
                self.current_netD_B_arch_diff_loss = \
                    (self.opt.ema_beta * self.criterionL1(self.loss_D_B_arch_fake_real, self.loss_D_B_arch_fake) +
                     (1.0 - self.opt.ema_beta) * self.current_netD_B_arch_diff_loss)
            else:
                self.current_netD_A_arch_diff_loss = self.criterionL1(self.loss_D_A_arch_fake_real,
                                                                      self.loss_D_A_arch_fake)
                self.current_netD_B_arch_diff_loss = self.criterionL1(self.loss_D_B_arch_fake_real,
                                                                      self.loss_D_B_arch_fake)
        else:
            self.current_netD_A_arch_diff_loss = self.criterionL1(self.loss_D_A_arch_fake_real,
                                                                  self.loss_D_A_arch_fake)
            self.current_netD_B_arch_diff_loss = self.criterionL1(self.loss_D_B_arch_fake_real,
                                                                  self.loss_D_B_arch_fake)

        return self.current_netD_A_arch_diff_loss, self.current_netD_B_arch_diff_loss

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B

        if self.opt.online_distillation or self.opt.normal_distillation:

            self.Tfake_A = self.teacher_model.fake_A
            self.Tfake_B = self.teacher_model.fake_B
            current_distillation_Afeatures = self.get_distillation_features(AorB='A')
            current_distillation_Bfeatures = self.get_distillation_features(AorB='B')

            self.teacher_model.netD_A(self.fake_B.detach())  # output teacher discriminator feature for netG_A
            self.teacher_model.netD_B(self.fake_A.detach())  # output teacher discriminator feature for netG_B
            teacher_discriminator_Afeatures = list(self.teacher_model.total_netD_A_features.values())
            current_distillation_Afeatures[-len(self.discriminator_extract_layers):] = teacher_discriminator_Afeatures[:]
            teacher_discriminator_Bfeatures = list(self.teacher_model.total_netD_B_features.values())
            current_distillation_Bfeatures[-len(self.discriminator_extract_layers):] = teacher_discriminator_Bfeatures[:]

            self.loss_content_A = 0.0
            self.loss_gram_A = 0.0
            self.loss_content_B = 0.0
            self.loss_gram_B = 0.0
            self.loss_L1_A = 0.0
            self.loss_L1_B = 0.0
            for i, feature in enumerate(current_distillation_Afeatures):

                if i < 4:  # generator feature should transform channel dimension consistent with teacher
                    feature = self.transform_A_convs[i](feature)
                target_feature = self.target_distillation_A_features[i]
                # import pdb
                # pdb.set_trace()
                # print(feature.size(), target_feature.size())
                self.loss_gram_A += self.criterionMSE(self.gram(feature), self.gram(target_feature.detach()))
                self.loss_content_A += self.criterionMSE(feature, target_feature.detach())
                self.loss_L1_A += self.criterionL1(self.fake_B, self.Tfake_B.detach())
            for i, feature in enumerate(current_distillation_Bfeatures):

                if i < 4:  # generator feature should transform channel dimension consistent with teacher
                    feature = self.transform_B_convs[i](feature)
                target_feature = self.target_distillation_B_features[i]
                self.loss_gram_B += self.criterionMSE(self.gram(feature), self.gram(target_feature.detach()))
                self.loss_content_B += self.criterionMSE(feature, target_feature.detach())
                self.loss_L1_B += self.criterionL1(self.fake_A, self.Tfake_A.detach())

            self.loss_gram_A = self.opt.lambda_gram * self.loss_gram_A
            self.loss_content_A = self.opt.lambda_content * self.loss_content_A
            self.loss_L1_A = self.opt.lambda_L1 * self.loss_L1_A
            self.loss_gram_B = self.opt.lambda_gram * self.loss_gram_B
            self.loss_content_B = self.opt.lambda_content * self.loss_content_B
            self.loss_L1_B = self.opt.lambda_L1 * self.loss_L1_B
            self.loss_G += self.loss_gram_A + self.loss_gram_B
            self.loss_G += self.loss_content_A + self.loss_content_B
            self.loss_G += self.loss_L1_A + self.loss_L1_B

        self.loss_G.backward()

        self.L1_sparsity()

    def L1_sparsity(self):

        if self.opt.lambda_weight > 0.0:
            for name, module in self.netG_A.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                    if name in self.heavy_sparsity:
                        if name == 'model.19':
                            module.weight.grad.data.add_(self.opt.lambda_weight * 1000 * torch.sign(module.weight.data))
                        else:
                            module.weight.grad.data.add_(self.opt.lambda_weight * 2 * torch.sign(module.weight.data))
                    else:
                        module.weight.grad.data.add_(self.opt.lambda_weight * torch.sign(module.weight.data))
            for name, module in self.netG_B.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                    if name in self.heavy_sparsity:
                        if name == 'model.19':
                            module.weight.grad.data.add_(self.opt.lambda_weight * 1000 * torch.sign(module.weight.data))
                        else:
                            module.weight.grad.data.add_(self.opt.lambda_weight * 2 * torch.sign(module.weight.data))
                    else:
                        module.weight.grad.data.add_(self.opt.lambda_weight * torch.sign(module.weight.data))

    def group_sparsity(self):

        if self.opt.lambda_weight > 0.0:
            pass


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        if self.opt.online_distillation:
            self.teacher_model.set_input(self.input)
            # else:
            self.teacher_model.optimize_parameters()
            self.target_distillation_A_features = [f.clone() for f in self.teacher_model.get_distillation_features(AorB='A')]
            self.target_distillation_B_features = [f.clone() for f in self.teacher_model.get_distillation_features(AorB='B')]
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.set_netD_arch_grad(False)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def optimizer_netD_arch(self):
        self.forward()
        self.teacher_model.set_input(self.input)
        self.teacher_model.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.set_netD_weight_grad(False)
        self.optimizer_arch.zero_grad()
        self.backward_D_arch()
        self.optimizer_arch.step()

    def print_sparse_info(self, logger):

        for name, module in self.netD_A.named_modules():
            if isinstance(module, DifferentiableOP):
                mask = module.get_current_mask()
                logger.info('netD_A %s sparsity ratio: %.2f' % (name, float(sum(mask == 0.0)) / mask.numel()))
        logger.info('-----------------------------------')
        for name, module in self.netD_B.named_modules():
            if isinstance(module, DifferentiableOP):
                mask = module.get_current_mask()
                logger.info('netD_B %s sparsity ratio: %.2f' % (name, float(sum(mask == 0.0)) / mask.numel()))

    def adaptive_ema_beta(self, epoch):

        self.opt.ema_beta = 1.0 - epoch / (self.opt.n_epochs + self.opt.n_epochs_decay)

    def update_learning_rate(self, epoch):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()

        self.adaptive_ema_beta(epoch)

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_netD_weight_grad(self, requires_grad=False):
        for module in self.netD_A.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d):
                for param in module.parameters():
                    param.requires_grad = requires_grad
        for module in self.netD_B.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d):
                for param in module.parameters():
                    param.requires_grad = requires_grad

    def set_netD_arch_grad(self, requires_grad=False):
        for module in self.netD_A.modules():
            if isinstance(module, DifferentiableOP):
                for param in module.parameters():
                    param.requires_grad = requires_grad
        for module in self.netD_B.modules():
            if isinstance(module, DifferentiableOP):
                for param in module.parameters():
                    param.requires_grad = requires_grad

    def save_models(self, epoch, save_dir, fid=None, isbest=False, direction='AtoB'):
        util.mkdirs(save_dir)
        ckpt = {
            'G_A': self.__pop_ops_params_state_dict(self.netG_A.state_dict()),
            'G_B': self.__pop_ops_params_state_dict(self.netG_B.state_dict()),
            'D_A': self.__pop_ops_params_state_dict(self.netD_A.state_dict()),
            'D_B': self.__pop_ops_params_state_dict(self.netD_B.state_dict()),
            'epoch': epoch,
            'cfg': (self.cfg_AtoB, self.cfg_BtoA),
            'fid': fid
        }
        if isbest:
            torch.save(ckpt, os.path.join(save_dir, 'model_best_%s.pth' % direction))
        else:
            torch.save(ckpt, os.path.join(save_dir, 'model_%d.pth' % epoch))

    def load_models(self, load_path, load_discriminator=True):
        ckpt = torch.load(load_path, map_location=self.device)
        self.netG_A.load_state_dict(self.__pop_ops_params_state_dict(ckpt['G_A']))
        self.netG_B.load_state_dict(self.__pop_ops_params_state_dict(ckpt['G_B']))
        if load_discriminator:
            self.netD_A.load_state_dict(self.__pop_ops_params_state_dict(ckpt['D_A']))
            self.netD_B.load_state_dict(self.__pop_ops_params_state_dict(ckpt['D_B']))

        print('loading the model from %s' % (load_path))

    def __pop_ops_params_state_dict(self, state_dict):

        for k in list(state_dict.keys()):
            if str.endswith(k, 'total_ops') or str.endswith(k, 'total_params'):
                state_dict.pop(k)
        return state_dict

    def init_net(self):
        self.netG_A.to(self.device)
        self.netG_B.to(self.device)
        self.netD_A.to(self.device)
        self.netD_B.to(self.device)

        if self.opt.darts_discriminator:

            for module in self.netD_A.modules():
                if isinstance(module, DifferentiableOP):
                    module.threshold = module.threshold.to(module.alpha.device)
            for module in self.netD_B.modules():
                if isinstance(module, DifferentiableOP):
                    module.threshold = module.threshold.to(module.alpha.device)

        util.init_weights(self.netG_A, init_type='normal', init_gain=0.02)
        util.init_weights(self.netG_B, init_type='normal', init_gain=0.02)
        util.init_weights(self.netD_A, init_type='normal', init_gain=0.02)
        util.init_weights(self.netD_B, init_type='normal', init_gain=0.02)

    def clipping_mask_alpha(self):
        for module in self.netD_A.modules():
            if isinstance(module, DifferentiableOP):
                module.clip_alpha()
        for module in self.netD_B.modules():
            if isinstance(module, DifferentiableOP):
                module.clip_alpha()

    def model_train(self):
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()

    def model_eval(self):
        self.netG_A.eval()
        self.netG_B.eval()
        self.netD_A.eval()
        self.netD_B.eval()

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

        self.total_netG_A_features = {}
        self.total_netG_B_features = {}
        self.total_netD_A_features = {}
        self.total_netD_B_features = {}

        if self.opt.online_distillation or self.opt.normal_distillation:
            if self.opt.lambda_content > 0.0:
                self.loss_names.append('content_A')
                self.loss_names.append('content_B')

            if self.opt.lambda_gram > 0.0:
                self.loss_names.append('gram_A')
                self.loss_names.append('gram_B')
            if self.opt.lambda_L1 > 0.0:
                self.loss_names.append('L1_A')
                self.loss_names.append('L1_B')
            self.visual_names.append('Tfake_A')
            self.visual_names.append('Tfake_B')

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name] = output
            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self.netG_A, self.total_netG_A_features, self.generator_extract_layers)
        add_hook(self.netG_B, self.total_netG_B_features, self.generator_extract_layers)
        add_hook(self.netD_A, self.total_netD_A_features, self.discriminator_extract_layers)
        add_hook(self.netD_B, self.total_netD_B_features, self.discriminator_extract_layers)

    def get_distillation_features(self, AorB='A'):
        if AorB == 'A':
            return list(self.total_netG_A_features.values()) + list(self.total_netD_A_features.values())
        else:
            return list(self.total_netG_B_features.values()) + list(self.total_netD_B_features.values())

    def gram(self, x):

        b, c, h, w = x.size()
        f = x.view(b, c, h * w)
        f_T = f.transpose(1, 2)

        G = f.bmm(f_T) / (c * h * w)
        return G

    def prune(self, threshold, lottery_path=None):

        return self.resnet_prune(threshold, lottery_path)

    def max_min_conv_norm(self, netG):

        overall_minnorm = float('inf')
        unprunable_layers_maxnorm = float('inf')
        prunable_layer_maxnorm = -float('inf')

        unprunable_layer_names = ['model.26']
        for i in range(10, 19, 1):
            unprunable_layer_names.append('model.%d.conv_block.1.conv.0' % i)
            unprunable_layer_names.append('model.%d.conv_block.6.conv.0' % i)

        residual_conv_names = ['model.7']
        for i in range(10, 19, 1):
            residual_conv_names.append('model.%d.conv_block.6.conv.2' % i)
        state_dict = netG.state_dict()
        residual_width = state_dict[residual_conv_names[0] + '.weight'].size(0)
        residual_norms = [0.0] * residual_width
        for name in residual_conv_names:
            weight_norm = torch.sum(torch.abs(state_dict[name + '.weight']), (1, 2, 3))
            for i in range(residual_width):
                residual_norms[i] += weight_norm[i]
        residual_mean_norm = torch.FloatTensor(residual_norms).to(self.device) / len(residual_conv_names)

        for name, module in netG.named_modules():

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):

                if name in unprunable_layer_names:
                    continue

                if isinstance(module, nn.Conv2d):
                    if name in residual_conv_names:
                        # continue
                        weight_norm = residual_mean_norm
                    else:
                        weight_norm = torch.sum(torch.abs(module.weight.data), (1, 2, 3))
                else:
                    weight_norm = torch.sum(torch.abs(module.weight.data), (0, 2, 3))

                unprunable_layers_maxnorm = min(torch.max(weight_norm), unprunable_layers_maxnorm)
                overall_minnorm = min(torch.min(weight_norm), overall_minnorm)
        return unprunable_layers_maxnorm, overall_minnorm

    def get_cfg(self):
        return self.cfg_AtoB, self.cfg_BtoA

    def get_prunenet_cfg(self, netG, threshold):

        filter_cfgs = []
        unprunable_layer_names = ['model.26']
        for i in range(10, 19, 1):
            unprunable_layer_names.append('model.%d.conv_block.1.conv.0' % i)
            unprunable_layer_names.append('model.%d.conv_block.6.conv.0' % i)

        residual_conv_names = ['model.7']
        for i in range(10, 19, 1):
            residual_conv_names.append('model.%d.conv_block.6.conv.2' % i)
        state_dict = netG.state_dict()
        residual_width = state_dict[residual_conv_names[0] + '.weight'].size(0)
        residual_norms = [0.0] * residual_width
        for name in residual_conv_names:
            weight_norm = torch.sum(torch.abs(state_dict[name + '.weight']), (1, 2, 3))
            for i in range(residual_width):
                residual_norms[i] += weight_norm[i]
        residual_mean_norm = torch.FloatTensor(residual_norms) / len(residual_conv_names)
        residual_mask = torch.FloatTensor(residual_mean_norm).to(self.device) > threshold

        for name, module in netG.named_modules():

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):

                if name in unprunable_layer_names:
                    continue

                if name in residual_conv_names:
                    # filter_cfgs.append(module.weight.data.size(0))
                    filter_cfgs.append(int(sum(residual_mask)))
                else:
                    weight = module.weight.data
                    if isinstance(module, nn.Conv2d):
                        weight_norm = torch.sum(torch.abs(weight), (1, 2, 3))
                    else:
                        weight_norm = torch.sum(torch.abs(weight), (0, 2, 3))
                    mask = weight_norm > threshold
                    filter_cfg = int(sum(mask))
                    filter_cfgs.append(filter_cfg)

        return filter_cfgs

    def resnet_prune(self, threshold_AtoB, threshold_BtoA):

        cfg_AtoB = self.get_prunenet_cfg(self.netG_A, threshold_AtoB)
        cfg_BtoA = self.get_prunenet_cfg(self.netG_B, threshold_BtoA)

        print(cfg_AtoB)
        print(cfg_BtoA)
        print('-------------------')

        pruned_model = MobileCycleGANModel(self.opt, cfg_AtoB=cfg_AtoB, cfg_BtoA=cfg_BtoA)

        return pruned_model