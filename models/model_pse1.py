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

inplace = True

class PSENet(nn.Module):
    def __init__(self, backbone, result_num=6, scale: int = 1, pretrained=False):
        super(PSENet, self).__init__()
        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        self.name = backbone
        self.scale = scale
        conv_out = 256
        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']
        self.backbone = backbone_model(pretrained=pretrained)

        # Top layer
        self.toplayer = nn.Conv2d(backbone_out[3], conv_out, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(backbone_out[2], conv_out, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(backbone_out[1], conv_out, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(backbone_out[0], conv_out, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(conv_out, conv_out, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(conv_out * 4, conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=inplace)
        )
        self.out_conv = nn.Conv2d(conv_out, result_num, kernel_size=1, stride=1)

    def forward(self, input: torch.Tensor):
        _, _, H, W = input.size()
        c2, c3, c4, c5 = self.backbone(input)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        x = self.out_conv(x)

        x = F.interpolate(x, size=(H // self.scale, W // self.scale), mode='bilinear', align_corners=True)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=False)
        return torch.cat([p2, p3, p4, p5], dim=1)

if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 640, 640).to(device)

    model = PAN(backbone='resnet18', fpem_repeat=2, pretrained=True).to(device)
    y = model(x)
    print(y.shape)
    # torch.save(model.state_dict(), 'PAN.pth')
