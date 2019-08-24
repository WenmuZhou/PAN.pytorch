# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:57
# @Author  : zhoujun

import torch
from torch import nn
import torch.nn.functional as F
from models.modules import *

backbone_dict = {'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
                 'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
                 'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
                 'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
                 'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
                 'resnext50_32x4d': {'models': resnext50_32x4d, 'out': [256, 512, 1024, 2048]},
                 'resnext101_32x8d': {'models': resnext101_32x8d, 'out': [256, 512, 1024, 2048]}
                 }


# 'MobileNetV3_Large': {'models': MobileNetV3_Large, 'out': [24, 40, 160, 160]},
# 'MobileNetV3_Small': {'models': MobileNetV3_Small, 'out': [16, 24, 48, 96]},
# 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}}


class PAN(nn.Module):
    def __init__(self, backbone, fpem_repeat, pretrained=False):
        """
        PANnet
        :param backbone: 基础网络
        :param fpem_repeat: FPEM模块重复的次数
        :param pretrained: 基础网络是否使用预训练模型
        """
        super().__init__()
        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']
        self.backbone = backbone_model(pretrained=pretrained)

        self.conv_c2 = nn.Conv2d(in_channels=backbone_out[0], out_channels=128, kernel_size=1)
        self.conv_c3 = nn.Conv2d(in_channels=backbone_out[1], out_channels=128, kernel_size=1)
        self.conv_c4 = nn.Conv2d(in_channels=backbone_out[2], out_channels=128, kernel_size=1)
        self.conv_c5 = nn.Conv2d(in_channels=backbone_out[3], out_channels=128, kernel_size=1)
        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(128))
        self.out_conv = nn.Conv2d(in_channels=512, out_channels=6, kernel_size=1)
        self.name = backbone

    def forward(self, x):
        _, _, H, W = x.size()
        c2, c3, c4, c5 = self.backbone(x)
        # reduce channel
        c2 = self.conv_c2(c2)
        c3 = self.conv_c3(c3)
        c4 = self.conv_c4(c4)
        c5 = self.conv_c5(c5)
        c2_ffm = c2
        c3_ffm = c3
        c4_ffm = c4
        c5_ffm = c5
        # FPEM
        for fpem in self.fpems:
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            c2_ffm += c2
            c3_ffm += c3
            c4_ffm += c4
            c5_ffm += c5

        # FFM
        c5 = F.interpolate(c5_ffm, c2_ffm.size()[-2:], mode='bilinear', align_corners=True)
        c4 = F.interpolate(c4_ffm, c2_ffm.size()[-2:], mode='bilinear', align_corners=True)
        c3 = F.interpolate(c3_ffm, c2_ffm.size()[-2:], mode='bilinear', align_corners=True)
        Fy = torch.cat([c2_ffm, c3, c4, c5], dim=1)
        y = self.out_conv(Fy)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


class FPEM(nn.Module):
    def __init__(self, in_channel=128):
        super().__init__()
        self.add_up = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.add_down = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel,
                      stride=2),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, c2, c3, c4, c5):
        # up阶段
        c4 = self.add_up(c4 + F.interpolate(c5, c4.size()[-2:], mode='bilinear', align_corners=True))
        c3 = self.add_up(c3 + F.interpolate(c4, c3.size()[-2:], mode='bilinear', align_corners=True))
        c2 = self.add_up(c2 + F.interpolate(c3, c2.size()[-2:], mode='bilinear', align_corners=True))

        # down 阶段
        c3 = self.add_down(c2 + F.interpolate(c3, c2.size()[-2:], mode='bilinear', align_corners=True))
        c4 = self.add_down(c3 + F.interpolate(c4, c3.size()[-2:], mode='bilinear', align_corners=True))
        c5 = self.add_down(c4 + F.interpolate(c5, c4.size()[-2:], mode='bilinear', align_corners=True))
        return c2, c3, c4, c5


if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 640, 640).to(device)

    model = PAN(backbone='resnet18', fpem_repeat=2, pretrained=True).to(device)
    y = model(x)
    print(y.shape)
    # torch.save(model.state_dict(), 'PAN.pth')
