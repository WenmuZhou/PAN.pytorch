# -*- coding: utf-8 -*-
# @Time    : 2019/9/8 14:18
# @Author  : zhoujun
import torch
import numpy as np


def decode_numpy(preds, label, label_values, dis_threshold=0.8):
    h, w = label.shape
    label = label.reshape(-1)
    text = preds[0].reshape(-1).astype(np.bool)
    similarity_vectors = preds[2:].reshape(4, -1)
    pred = np.zeros(text.shape)
    for i in label_values:
        kernel_idx = label == i
        pred[kernel_idx] = i
        kernel_similarity_vector = similarity_vectors[:, kernel_idx].mean(1)  # 4
        dis = np.linalg.norm(similarity_vectors - kernel_similarity_vector.reshape(4, 1), axis=0)
        dis[~text] = dis_threshold + 1  # 背景像素忽略
        idx = dis < dis_threshold
        pred[idx] = i
    return pred.reshape((h, w))


def decode_torch(preds, label, label_values, dis_threshold=0.8):
    h, w = label.shape
    label = label.reshape(-1)
    non_text = ~preds[0].reshape(-1).byte()
    similarity_vectors = preds[2:].reshape(4, -1)
    pred = torch.zeros(non_text.shape)
    for i in label_values:
        kernel_idx = label == i
        pred[kernel_idx] = i
        kernel_similarity_vector = similarity_vectors[:, kernel_idx].mean(1)  # 4
        dis = (similarity_vectors - kernel_similarity_vector.reshape(4, 1)).norm(2, dim=0)
        dis[non_text] = dis_threshold + 1  # 背景像素忽略
        idx = dis < dis_threshold
        pred[idx] = i
    return pred.reshape((h, w))
