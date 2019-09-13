# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:55
# @Author  : zhoujun
from .model import Model
from .loss import PANLoss


def get_model(config):
    model_config = config['arch']['args']
    return Model(model_config)

def get_loss(config):
    alpha = config['loss']['args']['alpha']
    beta = config['loss']['args']['beta']
    delta_agg = config['loss']['args']['delta_agg']
    delta_dis = config['loss']['args']['delta_dis']
    ohem_ratio = config['loss']['args']['ohem_ratio']
    return PANLoss(alpha=alpha, beta=beta, delta_agg=delta_agg, delta_dis=delta_dis, ohem_ratio=ohem_ratio)
