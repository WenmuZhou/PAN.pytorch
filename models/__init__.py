# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:55
# @Author  : zhoujun
from .model import PAN
from .model_pse1 import PSENet
from .loss import PANLoss


def get_model(config):
    backbone = config['arch']['args']['backbone']
    fpem_repeat = config['arch']['args']['fpem_repeat']
    pretrained = config['arch']['args']['pretrained']
    return PAN(backbone=backbone, fpem_repeat=fpem_repeat, pretrained=pretrained)

def get_model_pse1(config):
    backbone = config['arch']['args']['backbone']
    pretrained = config['arch']['args']['pretrained']
    return PSENet(backbone=backbone, pretrained=pretrained)

def get_loss(config):
    alpha = config['loss']['args']['alpha']
    beta = config['loss']['args']['beta']
    delta_agg = config['loss']['args']['delta_agg']
    delta_dis = config['loss']['args']['delta_dis']
    ohem_ratio = config['loss']['args']['ohem_ratio']
    return PANLoss(alpha=alpha, beta=beta, delta_agg=delta_agg, delta_dis=delta_dis, ohem_ratio=ohem_ratio)
