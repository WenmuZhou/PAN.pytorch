# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:58
# @Author  : zhoujun
import time
import torch
import torchvision.utils as vutils
from utils import PolynomialLR, runningScore, cal_text_score, cal_kernel_score

from base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, val_loader=None,
                 weights_init=None):
        super(Trainer, self).__init__(config, model, criterion, weights_init)
        self.show_images_interval = self.config['trainer']['show_images_interval']
        self.train_loader = train_loader
        self.train_loader_len = len(train_loader)
        self.val_loader = val_loader
        self.val_loader_len = len(val_loader) if val_loader is not None else 0

        # self.scheduler = PolynomialLR(self.optimizer, self.epochs * self.train_loader_len)

        self.logger.info(
            'train dataset has {} samples,{} in dataloader, val dataset has {} samples,{} in dataloader'.format(
                self.train_loader.dataset_len,
                self.train_loader_len,
                self.val_loader.dataset_len if val_loader is not None else 0,
                self.val_loader_len))

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        running_metric_text = runningScore(2)
        running_metric_kernel = runningScore(2)
        for i, (images, labels, training_masks) in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            # 数据进行转换和丢到gpu
            cur_batch_size = images.size()[0]
            images, labels, training_masks = images.to(self.device), labels.to(self.device), training_masks.to(
                self.device)

            preds = self.model(images)
            loss_all, loss_tex, loss_ker, loss_agg, loss_dis = self.criterion(preds, labels, training_masks)
            # backward
            self.optimizer.zero_grad()
            loss_all.backward()
            self.optimizer.step()
            # acc iou
            score_text = cal_text_score(preds[:, 0, :, :], labels[:, 0, :, :], training_masks, running_metric_text)
            score_kernel = cal_kernel_score(preds[:, 1, :, :], labels[:, 1, :, :], labels[:, 0, :, :], training_masks,
                                            running_metric_kernel)

            # loss 和 acc 记录到日志
            loss_all = loss_all.item()
            loss_tex = loss_tex.item()
            loss_ker = loss_ker.item()
            loss_agg = loss_agg.item()
            loss_dis = loss_dis.item()
            train_loss += loss_all
            acc = score_text['Mean Acc']
            iou_text = score_text['Mean IoU']
            iou_kernel = score_kernel['Mean IoU']

            if self.tensorboard_enable:
                # write tensorboard
                self.writer.add_scalar('TRAIN/LOSS/loss_all', loss_all, self.global_step)
                self.writer.add_scalar('TRAIN/LOSS/loss_tex', loss_tex, self.global_step)
                self.writer.add_scalar('TRAIN/LOSS/loss_ker', loss_ker, self.global_step)
                self.writer.add_scalar('TRAIN/LOSS/loss_agg', loss_agg, self.global_step)
                self.writer.add_scalar('TRAIN/LOSS/loss_dis', loss_dis, self.global_step)
                self.writer.add_scalar('TRAIN/ACC_IOU/acc', acc, self.global_step)
                self.writer.add_scalar('TRAIN/ACC_IOU/iou_text', iou_text, self.global_step)
                self.writer.add_scalar('TRAIN/ACC_IOU/iou_kernel', iou_kernel, self.global_step)
                self.writer.add_scalar('TRAIN/lr', lr, self.global_step)

            if (i + 1) % self.display_interval == 0:
                batch_time = time.time() - batch_start
                self.logger.info(
                    '[{}/{}], [{}/{}], global_step: {}, Speed: {:.1f} samples/sec, acc: {:.4f}, iou_text: {:.4f}, iou_kernel: {:.4f}, loss_all: {:.4f}, loss_tex: {:.4f}, loss_ker: {:.4f}, loss_agg: {:.4f}, loss_dis: {:.4f}, lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step,
                                            self.display_interval * cur_batch_size / batch_time, acc, iou_text,
                        iou_kernel, loss_all, loss_tex, loss_ker, loss_agg, loss_dis, lr, batch_time))
                batch_start = time.time()

            if i % self.show_images_interval == 0:
                # show images on tensorboard
                self.writer.add_images('TRAIN/imgs', images, self.global_step)
                # text kernel and training_masks
                gt_texts, gt_kernels = labels[:, 0, :, :], labels[:, 1, :, :]
                gt_texts[gt_texts <= 0.5] = 0
                gt_texts[gt_texts > 0.5] = 1
                gt_kernels[gt_kernels <= 0.5] = 0
                gt_kernels[gt_kernels > 0.5] = 1
                show_label = torch.cat([gt_texts, gt_kernels, training_masks.float()])
                show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=cur_batch_size, normalize=False, padding=20,
                                              pad_value=1)
                self.writer.add_image('TRAIN/gt', show_label, self.global_step)
                # model output
                preds[:, :2, :, :] = torch.sigmoid(preds[:, :2, :, :])
                show_pred = torch.cat([preds[:, 0, :, :], preds[:, 1, :, :]])
                show_pred = vutils.make_grid(show_pred.unsqueeze(1), nrow=cur_batch_size, normalize=False, padding=20,
                                             pad_value=1)
                self.writer.add_image('TRAIN/preds', show_pred, self.global_step)

        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': time.time() - epoch_start,
                'epoch': epoch}

    def _eval(self):
        pass

    def _on_epoch_finish(self):
        self.logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.epoch_result['lr']))

        save_best = False
        if self.val_loader is not None:
            epoch_eval_dict = self._eval()

            val_acc = epoch_eval_dict['n_correct'] / self.val_loader.dataset_len
            edit_dis = epoch_eval_dict['edit_dis'] / self.val_loader.dataset_len

            if self.tensorboard_enable:
                self.writer.add_scalar('EVAL/acc', val_acc, self.global_step)
                self.writer.add_scalar('EVAL/edit_distance', edit_dis, self.global_step)

            self.logger.info('[{}/{}], val_acc: {:.6f}'.format(self.epoch_result['epoch'], self.epochs, val_acc))

            net_save_path = '{}/CRNN_{}_loss{:.6f}_val_acc{:.6f}.pth'.format(self.checkpoint_dir,
                                                                             self.epoch_result['epoch'],
                                                                             self.epoch_result['train_loss'],
                                                                             val_acc)
            if val_acc > self.metrics['val_acc']:
                save_best = True
                self.metrics['val_acc'] = val_acc
                self.metrics['best_model'] = net_save_path
        else:
            net_save_path = '{}/CRNN_{}_loss{:.6f}.pth'.format(self.checkpoint_dir,
                                                               self.epoch_result['epoch'],
                                                               self.epoch_result['train_loss'])
            if self.epoch_result['train_loss'] < self.metrics['train_loss']:
                save_best = True
                self.metrics['train_loss'] = self.epoch_result['train_loss']
                self.metrics['best_model'] = net_save_path
        self._save_checkpoint(self.epoch_result['epoch'], net_save_path, save_best)

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger.info('{}:{}'.format(k, v))
        self.logger.info('finish train')
