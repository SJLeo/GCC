import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class STE_Function(torch.autograd.Function):

    @staticmethod
    def forward(self, input, alpha, threshold):
        mask = (alpha >= threshold).float()
        output = input * mask[None, :, None, None]
        self.save_for_backward(input, mask)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, mask = self.saved_tensors
        grad_input = grad_output * mask[None, :, None, None]
        grad_alpha = torch.sum(grad_output * input * mask[None, :, None, None], dim=(0, 2, 3))
        return grad_input, grad_alpha, None

class Mask(torch.autograd.Function):

    @staticmethod
    def forward(ctx, alpha, threshold):
        mask = (torch.sign(alpha - threshold) + 1) / 2
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class DifferentiableOP(nn.Module):

    def __init__(self, output_channel, threshold=0.5):
        super(DifferentiableOP, self).__init__()

        setattr(self, 'alpha', nn.Parameter(torch.ones(output_channel), requires_grad=True))
        setattr(self, 'threshold', threshold * torch.ones(1))

        # init.normal_(self.alpha.data, 0.5, 0.05)

    def forward(self, input):

        # output = STE_Function.apply(input, self.alpha, self.threshold)
        mask = Mask.apply(self.alpha, self.threshold)
        output = input * mask[None, :, None, None]
        return output

    def clip_alpha(self):

        self.alpha.data.clip_(0, 1)

    def anneal_threshold(self):
        pass

    def get_current_mask(self):
        return (torch.sign(self.alpha - self.threshold) + 1) / 2


# input = torch.randn((1, 3, 1, 1))
# target = torch.randn((1, 3, 1, 1))
# crite = nn.L1Loss()
# print(input)
# print(target)
# model = DifferentiableOP(3)
# output = model(input)
# print(output)
# loss = crite(output, target)
# loss.backward()
# print(model.get_current_mask())
# print(model.alpha.grad)

