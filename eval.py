# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import torch
import shutil
import numpy as np
import os
from tqdm import tqdm
from predict import Pytorch_model
from utils import cal_recall_precison_f1, draw_bbox

torch.backends.cudnn.benchmark = True


def main(model_path, img_folder, save_path, gpu_id):
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img_folder = os.path.join(save_path, 'img')
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    save_txt_folder = os.path.join(save_path, 'result')
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)
    img_paths = [os.path.join(img_folder, x) for x in os.listdir(img_folder)]
    model = Pytorch_model(model_path, gpu_id=gpu_id)
    total_frame = 0.0
    total_time = 0.0
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_txt_folder, 'res_' + img_name + '.txt')
        _, boxes_list, t = model.predict(img_path)
        total_frame += 1
        total_time += t
        # img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
        # cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), img)
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    print('fps:{}'.format(total_frame / total_time))
    return save_txt_folder


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str('2')
    scale = 4
    model_path = 'output/PAN_pred_mask_resnet50/checkpoint/model_best.pth'
    img_folder = '/data2/dataset/ICD15/test/img'
    gt_path = '/data2/dataset/ICD15/test/gt'
    save_path = model_path.replace('checkpoint/model_best.pt', 'result/')
    gpu_id = 0

    save_path = main(model_path, img_folder, save_path, gpu_id=gpu_id)
    result = cal_recall_precison_f1(gt_path=gt_path, result_path=save_path)
    print(result)
