import torch
import torch.nn as nn
import models.modules.blocks as B
import torch.nn.functional as F

import torch
import torch.nn as nn

class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)  #计算两个tensor的矩阵乘法
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class PAM_CAM_Layer(nn.Module):

    def __init__(self, in_ch, use_pam=True):
        super(PAM_CAM_Layer, self).__init__()

        self.attn = nn.Sequential(PAM_Module(in_ch) if use_pam else CAM_Module(in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.PReLU())

    def forward(self, x):
        return self.attn(x)

class LAEAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_recurs, upscale_factor, norm_type=None,
                 act_type='prelu'):
        super(LAEAN, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            projection_filter = 6
        if upscale_factor == 3:
            stride = 3
            padding = 2
            projection_filter = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            projection_filter = 8
        self.num_recurs = num_recurs  #12
        self.upscale_factor = upscale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fuse_channel = 8
        self.weight_M = nn.Parameter(torch.ones(self.num_recurs))
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = B.MeanShift(rgb_mean, rgb_std)
        self.conv_in = B.ConvBlock(in_channels, num_features, kernel_size=3, act_type=act_type, norm_type=None)

        self.first_Blocks = nn.ModuleList()
        for idx in range(self.num_recurs):
            self.first_Blocks.append(B.ConvBlock(num_features*(idx+1), num_features, kernel_size=1, act_type=None, norm_type=None))
        self.pam_attention_1_4 = PAM_CAM_Layer(num_features)
        self.cam_attention_1_4 = PAM_CAM_Layer(num_features, False)
        self.conv8_1 = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.conv_feat = B.ConvBlock((self.num_recurs+1)*num_features, num_features, kernel_size=1, act_type=act_type, norm_type=None)

        self.up1_1 = B.DeconvBlock(num_features, num_features, projection_filter, stride=stride,
                                   padding=padding, norm_type=None, act_type=act_type)
        self.up1_2 = B.DeconvBlock(num_features, num_features, projection_filter, stride=stride,
                                   padding=padding, norm_type=None, act_type=act_type)

        self.down1_1 = B.ConvBlock(num_features, num_features, projection_filter, stride=stride,
                                   padding=padding, norm_type=None, act_type=act_type)
        self.down1_2 = B.ConvBlock(num_features, num_features, projection_filter, stride=stride,
                                   padding=padding, norm_type=None, act_type=act_type)

        self.conv_fusion = B.ConvBlock(num_features, num_features, kernel_size=1, act_type=None, norm_type=None)
        self.deconv = B.DeconvBlock(num_features, num_features, projection_filter, stride=stride,
                                     padding=padding, norm_type=None, act_type=act_type)
        self.conv_out = B.ConvBlock(num_features, out_channels, kernel_size=1, act_type=None, norm_type=None)

        self.add_mean = B.MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x):
        weight_M = F.softmax(self.weight_M, 0)
        OUT_M = 0
        x = self.sub_mean(x)
        x = self.conv_in(x)  #特征提取   [3,3,64,64]
        residual = x
        x1 = x
        for i in range(3):
            out_up1 = self.up1_1(x1)
            out_down1 = self.down1_1(out_up1)
            out_up2 = self.up1_2(out_down1)
            res_up = abs(out_up2-out_up1)
            out_down2 = self.down1_2(res_up)
            x1 = out_down2 + x1

        for idx in range(self.num_recurs):
            attn_pam4 = self.pam_attention_1_4(x)
            attn_cam4 = self.cam_attention_1_4(x)
            attention1_4 = self.conv8_1((attn_cam4 + attn_pam4))
            x = F.sigmoid(attention1_4)* x1 + x
            OUT_M += x * weight_M[idx]

        compress = self.conv_fusion(OUT_M)  # [3,3,64,64]
        out = compress + residual
        out = self.deconv(out)
        out = self.conv_out(out)
        out = self.add_mean(out)
        return out


