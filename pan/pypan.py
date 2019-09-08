# -*- coding: utf-8 -*-
# @Time    : 2019/9/8 14:18
# @Author  : zhoujun
import numpy as np


def decode(preds, label, label_num, dis_threshold=6):
    text = preds[0]
    similarity_vectors = preds[2:]
    text_similarity_vector = similarity_vectors[:, text.astype(int)]
    pred = np.zeros(text.shape)
    for i in range(label_num):
        kernel_idx = label == i
        pred[kernel_idx] = i
        kernel_similarity_vector = similarity_vectors[:,kernel_idx].mean(1)  # 4
        dis = (text_similarity_vector - kernel_similarity_vector.reshape(4, 1)).norm(axis=1)
        idx = dis < dis_threshold
        pred[idx] = i
    return pred
