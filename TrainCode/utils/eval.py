# -*- coding: utf-8 -*-
"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/7/12 14:49
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/7/12 14:49
 *@Description: 指标评估
"""
import glob
import os

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength = self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis = 1) + self.hist.sum(axis = 0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        # mean acc
        accuracy = np.diag(self.hist).sum() / self.hist.sum()
        precision = np.nanmean(np.diag(self.hist) / self.hist.sum(axis = 1))
        recall = np.nanmean(np.diag(self.hist) / self.hist.sum(axis = 0))
        freq = self.hist.sum(axis = 1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        f1_score = 2 * precision * recall / (precision + recall)
        return accuracy, precision, recall, miou, fwavacc, f1_score


if __name__ == '__main__':
    mask_path = '../dataset/test/public/ISPRS2DSemanticLabelingContestVaihingen/20230919/masks'
    predict_path = '../results/public/ISPRS2DSemanticLabelingContestVaihingen/20230919/1m/build/build202305211733'
    pres = glob.glob(os.path.join(predict_path, '*.tif'))  # os.listdir(predict_path)
    masks = []
    predicts = []
    for im in pres:
        mask_name = im.split(predict_path)[-1].replace('/', '').replace('\\','')  # im.split('_mask')[0]+'.tif'   'road.tif'
        ma_path = os.path.join(mask_path, mask_name)
        mask = cv2.imread(ma_path, 0)
        pre = cv2.imread(im, 0)
        mask[mask > 0] = 1
        pre[pre > 0] = 1
        masks.append(mask)
        predicts.append(pre)

    el = IOUMetric(2)
    accuracy, precision, recall, miou, fwavacc, f1_score = el.evaluate(predicts, masks)
    print('accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)
    print('miou: ', miou)
    # print('iou: ', iou)
    print('fwavacc: ', fwavacc)
    print('f1_score: ', f1_score)