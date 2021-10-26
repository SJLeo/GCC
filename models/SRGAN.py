import torch.nn as nn
import torch

from thop import profile
from models.GANLoss import GANLoss, TruncatedVGG19
import utils.util as util
from collections import OrderedDict
from models.DifferentiableOp import DifferentiableOP
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from data.sr_dataset import convert_image

import os
import math

class ConvolutionalBlock(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None, mask=False, threshold=0.5):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channe;s
        :param kernel_size: kernel size
        :param stride: stride
        :param batch_norm: include a BN layer?
        :param activation: Type of activation; None if none
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # A container that will hold the layers in this convolutional block
        layers = list()

        # A convolutional layer
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # A batch normalization (BN) layer, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if mask:
            layers.append(DifferentiableOP(out_channels, threshold=threshold))

        # An activation layer, if wanted
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # Put together the convolutional block as a sequence of the layers in this container
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, in_channels, w, h)
        :return: output images, a tensor of size (N, out_channels, w, h)
        """
        output = self.conv_block(input)  # (N, out_channels, w, h)

        return output

class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output

class ResidualBlock(nn.Module):
    """
    A residual block, comprising two convolutional blocks with a residual connection across them.
    """

    def __init__(self, kernel_size=3, n_channels=64, inner_channels=None):
        """
        :param kernel_size: kernel size
        :param n_channels: number of input and output channels (same because the input must be added to the output)
        """
        super(ResidualBlock, self).__init__()

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels,
                                              out_channels=n_channels if inner_channels is None else inner_channels,
                                              kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels if inner_channels is None else inner_channels,
                                              out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        """
        Forward propagation.
        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: output images, a tensor of size (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output

class Generator(nn.Module):
    """
    The SRResNet, as defined in the paper.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4, filter_cfgs=None):
        """
        :param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        :param small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
        :param n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
        :param n_blocks: number of residual blocks
        :param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
        """
        super(Generator, self).__init__()

        # Scaling factor must be 2, 4, or 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels,
                                              kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        # A sequence of n_blocks residual blocks, each containing a skip-connection across the block
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels,
                            inner_channels=None if filter_cfgs is None else filter_cfgs[i]) for i in range(n_blocks)])

        # Another convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        # Upscaling is done by sub-pixel convolution, with each such block upscaling by a factor of 2
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels,
                                         scaling_factor=2) for i in range(n_subpixel_convolution_blocks)])

        # The last convolutional block
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        """
        Forward prop.
        :param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
        :return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)  # (N, 3, w, h)
        residual = output  # (N, n_channels, w, h)
        output = self.residual_blocks(output)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)
        output = self.subpixel_convolutional_blocks(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        sr_imgs = self.conv_block3(output)  # (N, 3, w * scaling factor, h * scaling factor)

        return sr_imgs

class Discriminator(nn.Module):
    """
    The discriminator in the SRGAN, as defined in the paper.
    """

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=4):
        """
        :param kernel_size: kernel size in all convolutional blocks
        :param n_channels: number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
        :param n_blocks: number of convolutional blocks
        :param fc_size: size of the first fully connected layer
        """
        super(Discriminator, self).__init__()

        in_channels = 3

        # A series of convolutional blocks
        # The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
        # The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
        # The first convolutional block is unique because it does not employ batch normalization
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i == 0 else in_channels * 2) if i % 2 == 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 == 0 else 2, batch_norm=i != 0, activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # An adaptive pool layer that resizes it to a standard size
        # For the default input size of 96 and 8 convolutional blocks, this will have no effect
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        #
        self.fc1 = nn.Linear(out_channels, 1)

    def forward(self, imgs):
        """
        Forward propagation.
        :param imgs: high-resolution or super-resolution images which must be classified as such, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))

        return output

class MaskDiscriminator(nn.Module):
    """
    The discriminator in the SRGAN, as defined in the paper.
    """

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=4, threshold=0.5):
        """
        :param kernel_size: kernel size in all convolutional blocks
        :param n_channels: number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
        :param n_blocks: number of convolutional blocks
        :param fc_size: size of the first fully connected layer
        """
        super(MaskDiscriminator, self).__init__()

        in_channels = 3

        # A series of convolutional blocks
        # The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
        # The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
        # The first convolutional block is unique because it does not employ batch normalization
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i == 0 else in_channels * 2) if i % 2 == 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 == 0 else 2, batch_norm=i != 0, activation='LeakyReLu', mask=True, threshold=threshold))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # An adaptive pool layer that resizes it to a standard size
        # For the default input size of 96 and 8 convolutional blocks, this will have no effect
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(out_channels, 1)

        # Don't need a sigmoid layer because the sigmoid operation is performed by PyTorch's nn.BCEWithLogitsLoss()

    def forward(self, imgs):
        """
        Forward propagation.
        :param imgs: high-resolution or super-resolution images which must be classified as such, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))

        return output

class SRGAN(nn.Module):

    def __init__(self, opt, filter_cfgs=None, channel_cfgs=None):
        super(SRGAN, self).__init__()

        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        self.filter_cfgs = filter_cfgs
        self.channel_cfgs = channel_cfgs
        self.current_epoch = 0
        self.optimizers = []
        self.current_D_arch_diff_loss = 0.0
        self.current_teacher_D_arch_fake_real = 0.0
        self.current_teacher_D_arch_fake = 0.0

        self.visual_names = ['real_lr', 'fake_hr', 'real_hr']
        if self.opt.generator_only:
            self.loss_names = ['content']
        else:
            self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'content', 'perceptual']

        self.generator_extract_layers = ['residual_blocks.3',
                                         'residual_blocks.7',
                                         'residual_blocks.11',
                                         'residual_blocks.15']
        if self.opt.darts_discriminator:
            self.discriminator_extract_layers = ['conv_blocks.1', 'conv_blocks.3']
        else:
            self.discriminator_extract_layers = ['conv_blocks.1', 'conv_blocks.3']

        self.netG = Generator(n_channels=self.opt.ngf, filter_cfgs=self.filter_cfgs)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.opt.lr)

        self.truncated_vgg19 = TruncatedVGG19(i=5, j=4)
        self.truncated_vgg19.eval()

        if self.opt.online_distillation or self.opt.normal_distillation:

            self.transform_convs = []

            teacher_extract_layer_ngf = [self.opt.teacher_ngf] * 4
            student_extract_layer_ngf = [self.opt.ngf] * 4
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
                        isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.ConvTranspose2d) or \
                        isinstance(module, nn.Linear):
                    netG_parms += list(module.parameters())
            self.optimizer_G = torch.optim.Adam(netG_parms, lr=self.opt.lr)

        if self.opt.darts_discriminator:
            self.loss_names.append('D_arch_diff')
            self.loss_names.append('D_arch')
            self.loss_names.append('teacher_D_arch_diff')
            self.netD = MaskDiscriminator(n_channels=opt.ndf, threshold=self.opt.threshold)
            weight_params = []
            arch_params = []
            for name, module in self.netD.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or \
                        isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.Linear):
                    weight_params += list(module.parameters())
                if isinstance(module, DifferentiableOP):
                    arch_params += list(module.parameters())
            self.optimizer_D = torch.optim.Adam(weight_params, lr=self.opt.lr)
            self.optimizer_arch = torch.optim.Adam(arch_params, lr=self.opt.arch_lr)
            if self.opt.arch_lr_step:
                self.optimizers.append(self.optimizer_arch)
        else:
            self.netD = Discriminator(n_channels=self.opt.ndf)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.opt.lr)
        if self.opt.generator_only:
            self.optimizers.clear()

        self.init_net()
        self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
        self.criterionL1 = nn.L1Loss().to(self.device)
        self.criterionMSE = nn.MSELoss().to(self.device)

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.schedulers = [util.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def set_input(self, input):

        self.input = input
        self.real_lr = input['lr'].to(self.device)
        self.real_hr = input['hr'].to(self.device)
        self.image_paths = [input['lr_names'], input['hr_names']]

    def forward(self):
        self.fake_hr = self.netG(self.real_lr)

    def backward_D(self):
        pred_real = self.netD(self.real_hr)
        pred_fake = self.netD(self.fake_hr.detach())

        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        self.loss_D = self.loss_D_real + self.loss_D_fake
        self.loss_D.backward()

    def backward_D_arch(self):

        self.loss_teacher_D_arch_diff, loss_teacher_D_arch_diff_sign = self.teacher_model.get_D_arch_diff(isTeacher=True)

        self.loss_D_arch_diff, loss_D_arch_diff_sign = self.get_D_arch_diff(isTeacher=False)
        self.loss_D_arch = self.criterionL1(self.loss_D_arch_diff, self.loss_teacher_D_arch_diff)
        self.loss_D_arch += self.loss_D_arch_real + self.loss_D_arch_fake
        self.loss_D_arch.backward()

    def get_D_arch_diff(self, isTeacher=False):

        self.real_hr = convert_image(self.real_hr, source='[-1, 1]', target='imagenet-norm')
        self.fake_hr = convert_image(self.fake_hr, source='[-1, 1]', target='imagenet-norm')

        if isTeacher:
            self.set_requires_grad(self.netD, False)  # teacher D requires no gradients when get loss_D_arch_diff

        pred_fake = self.netD(self.fake_hr.detach())
        self.loss_D_arch_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        self.loss_D_arch_fake_real = self.criterionGAN(pred_fake, True, for_discriminator=False)

        pred_real = self.netD(self.real_hr)
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

        self.loss_content = self.criterionMSE(self.fake_hr, self.real_hr) * self.opt.lambda_SR_content
        self.real_hr = convert_image(self.real_hr, source='[-1, 1]', target='imagenet-norm')
        self.fake_hr = convert_image(self.fake_hr, source='[-1, 1]', target='imagenet-norm')

        pred_fake = self.netD(self.fake_hr)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_SR_adversarial

        fake_hr_vgg = self.truncated_vgg19(self.fake_hr)
        real_hr_vgg = self.truncated_vgg19(self.real_hr).detach()
        self.loss_perceptual = self.criterionMSE(fake_hr_vgg, real_hr_vgg) * self.opt.lambda_SR_perceptual
        self.loss_G = self.loss_content + self.loss_G_GAN + self.loss_perceptual

        if self.opt.online_distillation or self.opt.normal_distillation:
            self.Tfake_hr = self.teacher_model.fake_hr
            current_distillation_features = self.get_distillation_features()
            self.teacher_model.netD(self.fake_hr) # output teacher discriminator feature for Gs
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
            self.loss_L1 = self.opt.lambda_L1 * self.criterionL1(self.fake_hr, self.Tfake_hr.detach())

            self.loss_G += self.loss_gram
            self.loss_G += self.loss_content
            self.loss_G += self.loss_L1

        self.loss_G.backward()

    def optimize_parameters(self):

        if self.opt.online_distillation:
            self.teacher_model.set_input(self.input)
            self.teacher_model.optimize_parameters()

            self.target_distillation_features = [f.clone() for f in self.teacher_model.get_distillation_features()]

        self.forward() # compute fake images: G(A)
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_netD_arch_grad(False) # disable backprop for differentiableOP
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

    def optimizer_netD_arch(self):
        self.forward()
        self.teacher_model.set_input(self.input)
        self.teacher_model.forward()
        self.set_requires_grad(self.netD, True)
        self.set_netD_weight_grad(False)
        self.optimizer_arch.zero_grad()
        self.backward_D_arch()
        self.optimizer_arch.step()

    def optimize_content_parameters(self):

        self.forward()  # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.loss_content = self.criterionMSE(self.fake_hr, self.real_hr)
        self.loss_content.backward()
        self.L1_sparsity()
        self.optimizer_G.step()  # udpate G's weights

    def L1_sparsity(self):

        if self.opt.lambda_weight > 0.0:
            for module in self.netG.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                    module.weight.grad.data.add_(self.opt.lambda_weight * torch.sign(module.weight.data))
        elif self.opt.lambda_scale > 0.0:
            for module in self.netG.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.weight.grad.data.add_(self.opt.lambda_scale * torch.sign(module.weight.data))

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

        self.current_epoch = epoch
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
        for module in self.netD.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) \
                    or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.Linear):
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
            'psnr': fid,
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
        return ckpt['psnr'], float('inf')

    def __pop_ops_params_state_dict(self, state_dict):

        for k in list(state_dict.keys()):
            if str.endswith(k, 'total_ops') or str.endswith(k, 'total_params'):
                state_dict.pop(k)
        return state_dict

    def init_net(self):
        self.netG.to(self.device)
        self.netD.to(self.device)
        self.truncated_vgg19.to(self.device)

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

    def get_current_psnr(self):
        self.fake_hr_y = convert_image(self.fake_hr, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        self.real_hr__y = convert_image(self.real_hr, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        psnr = peak_signal_noise_ratio(self.fake_hr_y.cpu().numpy(), self.real_hr__y.cpu().numpy(), data_range=255.)
        return psnr

    def get_current_ssim(self):
        ssim = structural_similarity(self.real_hr__y.cpu().numpy(), self.fake_hr_y.cpu().numpy(), data_range=255.)
        return ssim

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
            self.visual_names.append('Tfake_hr')

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

        if self.opt.scale_prune:
            return self.scale_prune(threshold, lottery_path)
        elif self.opt.norm_prune:
            return self.norm_prune(threshold, lottery_path)
        else:
            raise NotImplementedError('only scale and norm pruning are supported!!!')

    def max_min_bn_scale(self):

        prunable_layers_maxscale = float('inf')
        overall_minscale = float('inf')

        unprunable_layer_names = ['conv_block2.conv_block.1']
        for i in range(0, 16, 1):
            unprunable_layer_names.append('residual_blocks.15.conv_block2.conv_block.1')

        for name, module in self.netG.named_modules():

            if isinstance(module, nn.BatchNorm2d):

                if name in unprunable_layer_names:
                    continue
                else:
                    prunable_layers_maxscale = min(torch.max(module.weight.data), prunable_layers_maxscale)
                overall_minscale = min(torch.min(module.weight.data), overall_minscale)

        return prunable_layers_maxscale, overall_minscale

    def max_min_conv_norm(self):

        overall_minnorm = float('inf')
        prunable_layer_maxnorm = float('inf')

        unprunable_layer_names = ['conv_block3.0',
                                  'conv_block2.conv_block.0'
                                  'subpixel_convolutional_blocks.0.conv',
                                  'subpixel_convolutional_blocks.1.conv']

        for i in range(0, 16, 1):
            unprunable_layer_names.append('residual_blocks.15.conv_block2.conv_block.0')

        for name, module in self.netG.named_modules():

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):

                if name in unprunable_layer_names:
                    continue

                if isinstance(module, nn.Conv2d):
                    weight_norm = torch.sum(torch.abs(module.weight.data), (0, 2, 3))
                else:
                    weight_norm = torch.sum(torch.abs(module.weight.data), (1, 2, 3))

                prunable_layer_maxnorm = min(torch.max(weight_norm), prunable_layer_maxnorm)
                overall_minnorm = min(torch.min(weight_norm), overall_minnorm)

        return prunable_layer_maxnorm, overall_minnorm

    def get_cfg(self):
        return self.filter_cfgs, self.channel_cfgs

    def norm_prune(self, threshold, lottery_path=None):

        filter_cfgs = []
        lottery_masks = []  # Lottery theory

        unprunable_layer_names = ['conv_block1.conv_block.0', 'conv_block2.conv_block.0',
                                  'subpixel_convolutional_blocks.0.conv', 'subpixel_convolutional_blocks.1.conv']
        for i in range(0, 16, 1):
            unprunable_layer_names.append('residual_blocks.%d.conv_block2.conv_block.0' % i)

        for name, module in self.netG.named_modules():

            if isinstance(module, nn.Conv2d):

                if name in unprunable_layer_names:
                    continue
                else:
                    weight_norm = torch.sum(torch.abs(module.weight.data), (1, 2, 3))
                    mask = weight_norm > threshold
                    lottery_masks.append(mask)
                    filter_cfg = int(sum(mask))
                    filter_cfgs.append(filter_cfg)

        # print(filter_cfgs)
        # print(channel_cfgs)
        # print('-------------------')

        pruned_model = SRGAN(opt=self.opt, filter_cfgs=filter_cfgs)

        if lottery_path is not None:
            pruned_model.lottery_theory(lottery_masks, lottery_path)

        return pruned_model

    def scale_prune(self, threshold, lottery_path=None):

        filter_cfgs = []
        lottery_masks = []  # Lottery theory

        unprunable_layer_names = ['conv_block2.conv_block.1']
        for i in range(0, 16, 1):
            unprunable_layer_names.append('residual_blocks.%d.conv_block2.conv_block.1' % i)

        for name, module in self.netG.named_modules():

            if isinstance(module, nn.BatchNorm2d):

                if name in unprunable_layer_names:
                    continue
                else:
                    weight = module.weight.data
                    mask = weight > threshold
                    lottery_masks.append(mask)
                    filter_cfg = int(sum(mask))
                    filter_cfgs.append(filter_cfg)

        print(filter_cfgs)
        # print(channel_cfgs)
        print('-------------------')

        pruned_model = SRGAN(opt=self.opt, filter_cfgs=filter_cfgs)

        if lottery_path is not None:
            pruned_model.lottery_theory(lottery_masks, lottery_path)

        return pruned_model

# g = Generator(n_channels=64)
# macs, params = profile(g, (torch.randn(1, 3, 256, 256),), verbose=False)
# print(macs / (1000 ** 3), params / (1000 ** 2))
# print(g)

# d= Discriminator(n_blocks=4)
# from thop import profile
# macs, params = profile(d, (torch.randn(1, 3, 96, 96), ), verbose=False)
# print(macs / (1000 ** 3), (params / (1000 ** 2)))
# print(d)
