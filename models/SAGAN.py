import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from models.GANLoss import GANLoss
import utils.util as util
from models.DifferentiableOp import DifferentiableOP

import os
from collections import OrderedDict
import numpy as np

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class Generator(nn.Module):
    """Generator."""

    def __init__(self, ngf=64, image_size=64, z_dim=128, filter_cfgs=None):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num  # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, ngf * mult if filter_cfgs is None else filter_cfgs[0], 4)))
        layer1.append(nn.BatchNorm2d(ngf * mult if filter_cfgs is None else filter_cfgs[0]))
        layer1.append(nn.ReLU())

        curr_dim = ngf * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim if filter_cfgs is None else filter_cfgs[0],
                                                      int(curr_dim / 2) if filter_cfgs is None else filter_cfgs[1], 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2) if filter_cfgs is None else filter_cfgs[1]))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim if filter_cfgs is None else filter_cfgs[1],
                                                      int(curr_dim / 2) if filter_cfgs is None else filter_cfgs[2], 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2) if filter_cfgs is None else filter_cfgs[2]))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim if filter_cfgs is None else filter_cfgs[2],
                                                          int(curr_dim / 2) if filter_cfgs is None else filter_cfgs[3], 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2) if filter_cfgs is None else filter_cfgs[3]))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim if filter_cfgs is None else filter_cfgs[3], 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(ngf * 2 if filter_cfgs is None else filter_cfgs[2], 'relu')
        self.attn2 = Self_Attn(ngf if filter_cfgs is None else filter_cfgs[3], 'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        out = self.attn1(out)
        out = self.l4(out)
        out = self.attn2(out)
        out = self.last(out)

        return out

class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, ndf=64, image_size=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, ndf, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = ndf

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim * 2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(ndf * 4, 'relu')
        self.attn2 = Self_Attn(ndf * 8, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.attn1(out)
        out = self.l4(out)
        out = self.attn2(out)
        out = self.last(out)

        return out.squeeze()

class MaskDiscriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, ndf=64, image_size=64, threshold=0.5):
        super(MaskDiscriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, ndf, 4, 2, 1)))
        layer1.append(DifferentiableOP(ndf, threshold))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = ndf

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(DifferentiableOP(curr_dim * 2, threshold))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(DifferentiableOP(curr_dim * 2, threshold))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(DifferentiableOP(curr_dim * 2, threshold))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim * 2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(ndf * 4, 'relu')
        self.attn2 = Self_Attn(ndf * 8, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.attn1(out)
        out = self.l4(out)
        out = self.attn2(out)
        out = self.last(out)

        return out.squeeze()

class SAGANModel(nn.Module):

    def __init__(self, opt, filter_cfgs=None, channel_cfgs=None):
        super(SAGANModel, self).__init__()

        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        self.filter_cfgs = filter_cfgs
        self.channel_cfgs = channel_cfgs
        self.loss_names = ['G_GAN', 'D_real', 'D_fake']
        self.visual_names = ['fake_img', 'real_img']
        self.current_D_arch_diff_loss = 0.0
        self.current_teacher_D_arch_fake_real = 0.0
        self.current_teacher_D_arch_fake = 0.0

        self.generator_extract_layers = ['l2', 'attn2']
        if self.opt.darts_discriminator:
            self.discriminator_extract_layers = ['l2', 'attn2']
        else:
            self.discriminator_extract_layers = ['l2', 'attn2']

        self.optimizers = []
        self.netG = Generator(ngf=self.opt.ngf, image_size=self.opt.crop_size, z_dim=self.opt.z_dim, filter_cfgs=filter_cfgs)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0, 0.9))

        if self.opt.online_distillation or self.opt.normal_distillation:

            self.transform_convs = []

            teacher_extract_layer_ngf = [self.opt.teacher_ngf * 4, self.opt.teacher_ngf]
            if self.filter_cfgs is None:
                student_extract_layer_ngf = [self.opt.ngf * 4, self.opt.ngf]
            else:
                student_extract_layer_ngf = [self.filter_cfgs[1], self.filter_cfgs[3]]
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
                        isinstance(module, SpectralNorm) or isinstance(module, Self_Attn) or \
                        isinstance(module, nn.ConvTranspose2d):
                    netG_parms += list(module.parameters())
            self.optimizer_G = torch.optim.Adam(netG_parms, lr=self.opt.lr, betas=(0, 0.9))

        if self.opt.darts_discriminator:
            self.loss_names.append('D_arch_diff')
            self.loss_names.append('D_arch')
            self.loss_names.append('teacher_D_arch_diff')

            # if self.opt.regular:
            #     self.loss_names.append('regular')
            self.netD = MaskDiscriminator(ndf=opt.ndf)
            weight_params = []
            arch_params = []
            for name, module in self.netD.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or \
                        isinstance(module, SpectralNorm) or isinstance(module, Self_Attn) or \
                        isinstance(module, nn.ConvTranspose2d):
                    weight_params += list(module.parameters())
                if isinstance(module, DifferentiableOP):
                    arch_params += list(module.parameters())
            self.optimizer_D = torch.optim.Adam(weight_params, lr=self.opt.lr * 4, betas=(0, 0.9))
            self.optimizer_arch = torch.optim.Adam(arch_params, lr=self.opt.arch_lr)
            if self.opt.arch_lr_step:
                import copy
                arch_opt = copy.deepcopy(opt)
                arch_opt.lr_policy = 'step'
                arch_opt.lr_decay_iters = 40
                self.arch_scheduler = util.get_scheduler(self.optimizer_arch, arch_opt)
        else:
            self.netD = Discriminator(ndf=self.opt.ndf)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.opt.lr * 4, betas=(0, 0.9))

        self.init_net()

        self.criterionGAN = GANLoss(self.opt.gan_mode).to(self.device)
        self.criterionL1 = nn.L1Loss().to(self.device)
        self.criterionMSE = nn.MSELoss().to(self.device)

    def set_input(self, input):
        self.input = input
        self.z = input['z'].to(self.device)
        self.real_img = input['real_img'].to(self.device)
        self.image_paths = [input['img_path'], input['img_path']]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # print(self.z.size())
        self.fake_img = self.netG(self.z)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_img

        pred_real = self.netD(self.real_img)
        self.loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)

        pred_fake = self.netD(self.fake_img.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)

        self.loss_D = self.loss_D_fake + self.loss_D_real
        self.loss_D.backward()

    def backward_D_arch(self):

        self.loss_teacher_D_arch_diff, loss_teacher_D_arch_diff_sign = self.teacher_model.get_D_arch_diff(isTeacher=True)

        self.loss_D_arch_diff, loss_D_arch_diff_sign = self.get_D_arch_diff(isTeacher=False)
        self.loss_D_arch = self.criterionL1(self.loss_D_arch_diff, self.loss_teacher_D_arch_diff)
        self.loss_D_arch += self.loss_D_arch_real + self.loss_D_arch_fake
        self.loss_D_arch.backward()

    def get_D_arch_diff(self, isTeacher=False):

        if isTeacher:
            self.set_requires_grad(self.netD, False)  # teacher D requires no gradients when get loss_D_arch_diff

        pred_fake = self.netD(self.fake_img.detach())
        self.loss_D_arch_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        self.loss_D_arch_fake_real = self.criterionGAN(pred_fake, True, for_discriminator=False)

        pred_real = self.netD(self.real_img)
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

    def backward_new_D_arch(self):

        loss_teacher_D_fake_real, loss_teacher_D_fake = self.teacher_model.get_new_D_arch_diff(isTeacher=True)
        if self.current_teacher_D_arch_fake_real != 0:
            self.current_teacher_D_arch_fake_real = self.opt.ema_beta * loss_teacher_D_fake_real + \
                                                    (1.0 - self.opt.ema_beta) * self.current_teacher_D_arch_fake_real
            self.current_teacher_D_arch_fake = self.opt.ema_beta * loss_teacher_D_fake + \
                                               (1.0 - self.opt.ema_beta) * self.current_teacher_D_arch_fake
        else:
            self.current_teacher_D_arch_fake_real = loss_teacher_D_fake_real
            self.current_teacher_D_arch_fake = loss_teacher_D_fake

        self.get_new_D_arch_diff(isTeacher=False)
        self.loss_D_arch = self.loss_D_arch_real + self.loss_D_arch_fake
        self.loss_D_arch_diff = self.criterionL1((self.loss_D_arch_fake_real - self.current_teacher_D_arch_fake_real),
                                            (self.loss_D_arch_fake - self.current_teacher_D_arch_fake))
        self.loss_D_arch += self.loss_D_arch_diff
        self.loss_D_arch.backward()

    def get_new_D_arch_diff(self, isTeacher=False):

        if isTeacher:
            self.set_requires_grad(self.netD, False)  # teacher D requires no gradients when get loss_D_arch_diff

        pred_fake = self.netD(self.fake_img.detach())
        self.loss_D_arch_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        self.loss_D_arch_fake_real = self.criterionGAN(pred_fake, True, for_discriminator=False)

        pred_real = self.netD(self.real_img)
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

        pred_fake = self.netD(self.fake_img)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True, for_discriminator=False)

        self.loss_G = self.loss_G_GAN

        if self.opt.online_distillation or self.opt.normal_distillation:
            self.Tfake_img = self.teacher_model.fake_img
            current_distillation_features = self.get_distillation_features()
            self.teacher_model.netD(self.fake_img) # output teacher discriminator feature for Gs
            teacher_discriminator_features = list(self.teacher_model.total_discriminator_features.values())
            current_distillation_features[-len(self.discriminator_extract_layers):] = teacher_discriminator_features[:]

            self.loss_content = 0.0
            self.loss_gram = 0.0
            for i, feature in enumerate(current_distillation_features):

                if i < 2: # generator feature should transform channel dimension consistent with teacher
                    feature = self.transform_convs[i](feature)
                target_feature = self.target_distillation_features[i]
                self.loss_gram += torch.sqrt(self.criterionMSE(self.gram(feature), self.gram(target_feature.detach())))
                self.loss_content += torch.sqrt(self.criterionMSE(feature, target_feature.detach()))

            self.loss_gram = self.opt.lambda_gram * self.loss_gram
            self.loss_content = self.opt.lambda_content * self.loss_content
            self.loss_L1 = self.opt.lambda_L1 * self.criterionL1(self.fake_img, self.Tfake_img.detach())

            self.loss_G += self.loss_gram
            self.loss_G += self.loss_content
            self.loss_G += self.loss_L1

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
        if self.opt.arch_lr_step:
            self.arch_scheduler.step()

        self.adaptive_ema_beta(epoch)

        lr = self.optimizer_G.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

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
            'G': self.__pop_ops_params_state_dict(self.netG.state_dict()),
            'D': self.__pop_ops_params_state_dict(self.netD.state_dict()),
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
        self.netG.load_state_dict(self.__pop_ops_params_state_dict(ckpt['G']))
        if load_discriminator:
            self.netD.load_state_dict(self.__pop_ops_params_state_dict(ckpt['D']))

        print('loading the model from %s' % (load_path))
        return ckpt['fid'], float('inf')

    def __pop_ops_params_state_dict(self, state_dict):

        for k in list(state_dict.keys()):
            if str.endswith(k, 'total_ops') or str.endswith(k, 'total_params'):
                state_dict.pop(k)
        return state_dict

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
            if self.opt.lambda_L1 > 0.0:
                self.loss_names.append('L1')
            self.visual_names.append('Tfake_img')

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

    def prune(self, threshold, lottery_path=None):

        if self.opt.backbone == 'resnet':
            return self.resnet_prune(threshold, lottery_path)
        else:
            if self.opt.scale_prune:
                return self.scale_prune(threshold)
            elif self.opt.norm_prune:
                return self.norm_prune(threshold, lottery_path)
            else:
                raise NotImplementedError('only scale and norm pruning are supported!!!')

    def max_min_bn_scale(self):


        unprunable_layers_maxscale = float('inf')
        overall_minscale = float('inf')

        for name, module in self.netG.named_modules():

            if isinstance(module, nn.BatchNorm2d):

                unprunable_layers_maxscale = min(torch.max(module.weight.data), unprunable_layers_maxscale)
                overall_minscale = min(torch.min(module.weight.data), overall_minscale)

        return unprunable_layers_maxscale, overall_minscale

    def max_min_conv_norm(self):

        pass

    def get_cfg(self):
        return self.filter_cfgs, self.channel_cfgs

    def scale_prune(self, threshold):

        filter_cfgs = {
            'l1':0,
            'l2':0,
            'l3':0,
            'l4':0
        }

        for name, module in self.netG.named_modules():

            if isinstance(module, nn.BatchNorm2d):

                bn_weight = module.weight.data
                mask = bn_weight > threshold
                filter_cfg = int(sum(mask))
                # filter_cfgs.append(filter_cfg)
                filter_cfgs[name.split('.')[0]] = filter_cfg
        filter_cfgs = list(filter_cfgs.values())
        print(filter_cfgs)
        print('-----------------------')

        pruned_model = SAGANModel(self.opt, filter_cfgs=filter_cfgs)

        return pruned_model

    def norm_prune(self, threshold, lottery_path=None):

        pass


# g = Generator(ngf=48)
# print(g)
# from thop import profile
# macs, params = profile(g, (torch.randn(1, 128), ), verbose=False)
# # # # # print(g)
# print(macs / (1000 ** 2), params / (1000 ** 2))

# generator_extract_layers = ['l2', 'attn2']
# discriminator_extract_layers = ['l2', 'attn2']