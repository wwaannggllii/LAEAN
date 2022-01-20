import torch
from torch import nn
from torchvision.models.vgg import vgg16
import PerceptualSimilarity.models.util as ps
import random
import torch.nn.functional as F
from .gather import GatherLayer

# import pytorch_ssim

class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=False):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img

class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)

def discriminator_loss(reals, fakes, wasserstein=False, grad_penalties=None, weights=None):
    if not isinstance(reals, list):
        reals = (reals,)
    if not isinstance(fakes, list):
        fakes = (fakes,)
    if weights is None:
        weights = [1.0 / len(fakes)] * len(fakes)
    loss = 0.0
    if wasserstein:
        if not isinstance(grad_penalties, list):
            grad_penalties = (grad_penalties,)
        for real, fake, weight, grad_penalty in zip(reals, fakes, weights, grad_penalties):
            loss += weight * (-real.mean() + fake.mean() + grad_penalty)
    else:
        for real, fake, weight in zip(reals, fakes, weights):
            loss += weight * (-torch.log(real + 1e-8).mean() - torch.log(1 - fake + 1e-8).mean())
    return loss

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss

######################################################################################################################
def generator_loss(labels, wasserstein=False, weights=None):
    if not isinstance(labels, list):
        labels = (labels,)
    if weights is None:
        weights = [1.0 / len(labels)] * len(labels)
    loss = 0.0
    for label, weight in zip(labels, weights):
        if wasserstein:
            loss += weight * torch.mean(-label)
        else:
            loss += weight * torch.mean(-torch.log(label + 1e-8))
    return loss


class PerceptualLossVGG16(nn.Module):
    def __init__(self):
        super(PerceptualLossVGG16, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):
        return self.mse_loss(self.loss_network(x), self.loss_network(y))

class PerceptualLossLPIPS(nn.Module):
    def __init__(self):
        super(PerceptualLossLPIPS, self).__init__()
        self.loss_network = ps.PerceptualLoss(use_gpu=torch.cuda.is_available())

    def forward(self, x, y):
        return self.loss_network.forward(x, y, normalize=True).mean()

class PerceptualLoss(nn.Module):
    def __init__(self, rotations=False, flips=False):
        super(PerceptualLoss, self).__init__()
        self.loss = PerceptualLossLPIPS()
        self.rotations = rotations
        self.flips = flips

    def forward(self, x, y):
        if self.rotations:
            k_rot = random.choice([-1, 0, 1])
            x = torch.rot90(x, k_rot, [2, 3])
            y = torch.rot90(y, k_rot, [2, 3])
        if self.flips:
            if random.choice([True, False]):
                x = torch.flip(x, (2,))
                y = torch.flip(y, (2,))
            if random.choice([True, False]):
                x = torch.flip(x, (3,))
                y = torch.flip(y, (3,))
        return self.loss(x, y)

class GeneratorLossW(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, use_perceptual_loss=True, wgan=False, w_col=1,
                 w_tex=0.001, w_per=0.1, gaussian=False, lpips_rot_flip=False, **kwargs):
        super(GeneratorLossW, self).__init__()
        self.pixel_loss = nn.L1Loss()
        self.per_type = 'LPIPS'  #selecting different Perceptual loss type



        #gaussian=True 高斯滤波，gaussian=False 池化滤波
        self.color_filter = FilterLow(recursions=recursions, stride=stride, kernel_size=kernel_size, padding=False,
                                      gaussian=False)

        if torch.cuda.is_available():
            self.pixel_loss = self.pixel_loss.cuda()

        if self.per_type == 'LPIPS':
            self.perceptual_loss = PerceptualLoss(rotations=lpips_rot_flip, flips=lpips_rot_flip)
        elif self.per_type == 'VGG':
            self.perceptual_loss = PerceptualLossVGG16()
        else:
            raise NotImplemented('{} is not recognized'.format(self.per_type))

        self.use_perceptual_loss = use_perceptual_loss
        self.wasserstein = True
        self.w_col = w_col
        self.w_tex = w_tex
        self.w_per = w_per
        self.last_tex_loss = 0
        self.last_per_loss = 0
        self.last_col_loss = 0
        self.gaussian = gaussian
        self.last_mean_loss = 0

    def forward(self, tex_labels, out_images, target_images):
        # Adversarial Texture Loss
        self.last_tex_loss = generator_loss(tex_labels, wasserstein=self.wasserstein)
        # Perception Loss
        self.last_per_loss = self.perceptual_loss(out_images, target_images)
        # Color Loss
        self.last_col_loss = self.color_loss(out_images, target_images)
        loss = self.w_col * self.last_col_loss + self.w_tex * self.last_tex_loss
        if self.use_perceptual_loss:
            loss += self.w_per * self.last_per_loss
        return loss

    def color_loss(self, x, y):
        return self.pixel_loss(self.color_filter(x), self.color_filter(y))

    def rgb_loss(self, x, y):
        return self.pixel_loss(x.mean(3).mean(2), y.mean(3).mean(2))

    def mean_loss(self, x, y):
        return self.pixel_loss(x.view(x.size(0), -1).mean(1), y.view(y.size(0), -1).mean(1))

class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        vgg = vgg16(pretrained=True)
        self.loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, y, target, fake_label=None):
        total_loss, losses = self.image_restoration(y, target)
        if fake_label:
            total_loss += fake_label
        return total_loss, losses

    def image_restoration(self, pred, target):
        perceptual_loss = self.perceptual_loss(pred, target)
        l1 = F.l1_loss(pred, target)
        ssim_loss = 1 - self.ssim_loss(pred, target)
        del pred, target
        Loss = perceptual_loss + l1 + ssim_loss

        return Loss, (perceptual_loss, l1, ssim_loss)

    def  perceptual_loss(self, out_images, target_images):

        loss = self.mse_loss(
            self.loss_network(out_images),
            self.loss_network(target_images)) / 3
        return loss

    def CharbonnierLoss(self, y, target, eps=1e-6):
        diff = y - target
        loss = torch.mean(torch.sqrt(diff * diff + eps))
        return loss

    def tv_loss(self, x, TVLoss_weight=1):
        def _tensor_size(t):
            return t.size()[1] * t.size()[2] * t.size()[3]

        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = _tensor_size(x[:, :, 1:, :])
        count_w = _tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return TVLoss_weight * 2 * (
            h_tv / count_h + w_tv / count_w) / batch_size

class ms_Loss(Loss):
    def __init__(self):
        super(ms_Loss, self).__init__()
        self.L1_loss = nn.L1Loss()

    def forward(self, y, target, texture_img=None):
        loss = 0
        total_l1 = 0
        total_perceptual = 0
        total_ssim = 0
        # scale 1
        if texture_img:
            l1 = self.CharbonnierLoss(texture_img, target) * 0.25
            total_l1 += l1
            loss += l1
        # for i in range(len(y)):
        #     if i == 0:
        #         perceptual_loss = self.perceptual_loss(y[i], target)
        #         ssim_loss = 1 - self.ssim_loss(y[i], target)
        #         l1 = self.CharbonnierLoss(y[i], target)
        #         loss += 0.05 * ssim_loss + l1 + 0.25 * perceptual_loss
        #         total_l1 += l1
        #         total_perceptual += perceptual_loss
        #         total_ssim += ssim_loss
        #     elif i == 1 or i == 2:
        #         h, w = y[i].size(2), y[i].size(3)
        #         target = F.interpolate(target, size=(h, w))
        #         perceptual_loss = self.perceptual_loss(y[i], target)
        #         l1 = F.smooth_l1_loss(y[i], target)
        #         l1 = self.CharbonnierLoss(y[i], target)
        #         total_l1 += l1
        #         loss += perceptual_loss + l1 * 0.25
        #         total_perceptual += perceptual_loss
        #     else:
        #         h, w = y[i].size(2), y[i].size(3)
        #         target = F.interpolate(target, size=(h, w))
        #         l1 = self.CharbonnierLoss(y[i], target)
        #         total_l1 += l1
        #         loss += l1

        # print(y.shape)
        # print(target.shape)
        # exit()
        perceptual_loss = self.perceptual_loss(y, target)
        ssim_loss = 1 - self.ssim_loss(y, target)
        # l1 = self.CharbonnierLoss(y, target)
        l1 = self.L1_loss(y, target)
        loss += 0.05 * ssim_loss + l1 + 0.25 * perceptual_loss
        total_l1 += l1
        total_perceptual += perceptual_loss
        total_ssim += ssim_loss

        return loss


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]





if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
