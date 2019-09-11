# -*- coding: utf-8 -*-
# @Time    : 2019/9/11 16:24
# @Author  : zhoujun

import numpy as np
from sklearn.cluster import KMeans

def km(text, similarity_vectors, label, label_values, dis_threshold=0.8):
    similarity_vectors = similarity_vectors * np.expand_dims(text,2)
    # 计算聚类中心
    cluster_centers = [[0,0,0,0]]
    for i in label_values:
        kernel_idx = label == i
        kernel_similarity_vector = similarity_vectors[kernel_idx].mean(0)  # 4
        cluster_centers.append(kernel_similarity_vector)
    n = len(label_values) + 1
    similarity_vectors = similarity_vectors.reshape(-1,4)
    y_pred = KMeans(n,init=np.array(cluster_centers),n_init=1).fit_predict(similarity_vectors)
    y_pred = y_pred.reshape(text.shape)
    return y_pred
