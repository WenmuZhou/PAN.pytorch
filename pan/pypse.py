# -*- coding: utf-8 -*-
# @Time    : 2019/9/9 17:23
# @Author  : zhoujun

import numpy as np
from queue import Queue


def get_dis(sv1, sv2):
    return np.linalg.norm(sv1 - sv2)


def pse(text, similarity_vectors, label, label_values, dis_threshold=0.8):
    pred = np.zeros(text.shape)
    queue = Queue(maxsize=0)
    points = np.array(np.where(label > 0)).transpose((1, 0))

    for point_idx in range(points.shape[0]):
        x, y = points[point_idx, 0], points[point_idx, 1]
        l = label[x, y]
        queue.put((x, y, l))
        pred[x, y] = l
    # 计算kernel的值
    d = {}
    for i in label_values:
        kernel_idx = label == i
        kernel_similarity_vector = similarity_vectors[kernel_idx].mean(0)  # 4
        d[i] = kernel_similarity_vector

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    kernal = text.copy()
    while not queue.empty():
        (x, y, l) = queue.get()
        cur_kernel_sv = d[l]
        for j in range(4):
            tmpx = x + dx[j]
            tmpy = y + dy[j]
            if tmpx < 0 or tmpx >= kernal.shape[0] or tmpy < 0 or tmpy >= kernal.shape[1]:
                continue
            if kernal[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                continue
            if np.linalg.norm(similarity_vectors[x, y] - cur_kernel_sv) >= dis_threshold:
                continue
            queue.put((tmpx, tmpy, l))
            pred[tmpx, tmpy] = l
    return pred
