"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/7/12 14:49
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/7/12 14:49
 *@Description: 数据处理
"""
import os
import traceback

import cv2
import numpy as np
import torch
import torch.utils.data as data
import albumentations as aug
import random as rd

from skimage.exposure import match_histograms


def random_hue_saturation_value(image, hue_shift_limit = (-180, 180),
                                sat_shift_limit = (-255, 255),
                                val_shift_limit = (-255, 255), u = 0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def random_shift_scale_rotate(image, mask,
                              shift_limit = (-0.0, 0.0),
                              scale_limit = (-0.0, 0.0),
                              rotate_limit = (-0.0, 0.0),
                              aspect_limit = (-0.0, 0.0),
                              borderMode = cv2.BORDER_CONSTANT, u = 0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags = cv2.INTER_LINEAR, borderMode = borderMode,
                                    borderValue = (
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags = cv2.INTER_LINEAR, borderMode = borderMode,
                                   borderValue = (
                                       0, 0,
                                       0,))

    return image, mask


def random_horizontal_flip(image, mask, u = 0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def random_verticle_flip(image, mask, u = 0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def random_rotate90(image, mask, u = 0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


# 随机直方图拉伸
def radiation_random(img_copy, img_target):
    # 先随机直方图
    img_target = match_histograms(img_target, img_copy, multichannel = True)

    max_r = np.max(img_target[:, :, 0])
    min_r = np.min(img_target[:, :, 0])
    max_g = np.max(img_target[:, :, 1])
    min_g = np.min(img_target[:, :, 1])
    max_b = np.max(img_target[:, :, 2])
    min_b = np.min(img_target[:, :, 2])
    #
    min_rr = np.random.randint(0, 70)
    max_rr = np.random.randint(min_rr * 2, 255)
    min_gg = np.random.randint(0, 70)
    max_gg = np.random.randint(min_gg * 2, 255)
    min_bb = np.random.randint(0, 70)
    max_bb = np.random.randint(min_bb * 2, 255)

    img_target[:, :, 0] = (img_target[:, :, 0] - min_r) / (max_r - min_r) * (max_rr - min_rr) + min_rr
    img_target[:, :, 1] = (img_target[:, :, 1] - min_g) / (max_g - min_g) * (max_gg - min_gg) + min_gg
    img_target[:, :, 2] = (img_target[:, :, 2] - min_b) / (max_b - min_b) * (max_bb - min_bb) + min_bb

    img_target = np.array(img_target).astype("uint8")

    return img_target


# 限制直方图均衡化
def limit_histogram_equalization(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    clahe_b = clahe.apply(b)
    clahe_g = clahe.apply(g)
    clahe_r = clahe.apply(r)
    clahe_merge = cv2.merge((clahe_b, clahe_g, clahe_r))
    return clahe_merge


# Albumentations
def strong_aug(p = .5):
    return aug.Compose([
        aug.OneOf([aug.RandomRotate90(),
                   aug.ShiftScaleRotate(shift_limit = 0.0625, scale_limit = 0.2, rotate_limit = 45,
                                        p = 1 if rd.random() > 0.5 else 1e-8),
                   aug.Rotate((90, 225), p = 1 if rd.random() > 0.5 else 1e-8)], p = 1 if rd.random() > 0.5 else 1e-8),
        aug.OneOf([aug.HorizontalFlip(), aug.VerticalFlip()], p = 1 if rd.random() > 0.5 else 1e-8),
        aug.Transpose(),
        aug.GaussNoise(p = 1 if rd.random() > 0.5 else 1e-8),
        aug.OneOf([
            aug.MotionBlur(p = 1 if rd.random() > 0.5 else 1e-8),
            aug.MedianBlur(blur_limit = 3, p = 1 if rd.random() > 0.5 else 1e-8),
            aug.Blur(blur_limit = 3, p = 1 if rd.random() > 0.5 else 1e-8),
        ], p = 1 if rd.random() > 0.5 else 1e-8),
        aug.OneOf([
            aug.OpticalDistortion(p = 1 if rd.random() > 0.5 else 1e-8),
            aug.GridDistortion(p = 1 if rd.random() > 0.5 else 1e-8),
            aug.PiecewiseAffine(p = 1 if rd.random() > 0.5 else 1e-8),
        ], p = 1 if rd.random() > 0.5 else 1e-8),
        aug.OneOf([
            aug.CLAHE(clip_limit = 2, p = 1 if rd.random() > 0.5 else 1e-8),
            aug.Sharpen(p = 1 if rd.random() > 0.5 else 1e-8),
            aug.Emboss(p = 1 if rd.random() > 0.5 else 1e-8),
            aug.RandomBrightnessContrast(p = 1 if rd.random() > 0.5 else 1e-8),
            aug.HueSaturationValue(hue_shift_limit = 20, sat_shift_limit = 50, val_shift_limit = 50,
                                   p = 1 if rd.random() > 0.5 else 1e-8),
            aug.ChannelShuffle(p = 1 if rd.random() > 0.5 else 1e-8),
            aug.RandomGamma(p = 1 if rd.random() > 0.5 else 1e-8),
        ], p = 1 if rd.random() > 0.5 else 1e-8),
        aug.HueSaturationValue(p = 1 if rd.random() > 0.5 else 1e-8),
        aug.OpticalDistortion(p = 1 if rd.random() > 0.5 else 1e-8, distort_limit = 2, shift_limit = 0.5),
        aug.GridDistortion(p = 1 if rd.random() > 0.5 else 1e-8),
        aug.ElasticTransform(p = 1 if rd.random() > 0.5 else 1e-8, alpha = 120, sigma = 120 * 0.05, alpha_affine = 120 * 0.03)
    ], p = p)


def default_loader(id, root, img_type, label_type, copy_id):
    img = cv2.imread(os.path.join(root + 'imgs/' + '{}' + img_type).format(
        id))  # cv2.imdecode(np.fromfile(os.path.join(root + 'imgs/' + '{}' + img_type).format(id), dtype=np.uint8),-1)
    # img_copy = cv2.imread(os.path.join(root + 'imgs/' + '{}' + img_type).format(
    #     copy_id))
    mask = cv2.imread(os.path.join(root + 'masks/' + '{}_mask' + label_type).format(id), cv2.IMREAD_GRAYSCALE)
    # cv2.imdecode(np.fromfile(os.path.join(root + 'masks/' + '{}_mask' + label_type).format(id),dtype=np.uint8),-1)
    # augmentation = strong_aug(p=1)
    # augmented = augmentation(image=img, mask=mask)
    # img, mask = augmented['image'], augmented['mask']
    try:
        img = random_hue_saturation_value(img,
                                       hue_shift_limit=(-30, 30),
                                       sat_shift_limit=(-5, 5),
                                       val_shift_limit=(-15, 15))
        # img = radiation_random(img_copy, img)
        # img = limit_histogram_equalization(img)

        img, mask = random_shift_scale_rotate(img, mask, shift_limit = (-0.1, 0.1), scale_limit = (-0.1, 0.1),
                                              rotate_limit = (-0.0, 0.0), aspect_limit = (-0.1, 0.1))
        img, mask = random_horizontal_flip(img, mask)
        img, mask = random_verticle_flip(img, mask)
        img, mask = random_rotate90(img, mask)

        mask = np.expand_dims(mask, axis = 2)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        # mask = abs(mask-1)
    except:
        print(id, copy_id)

    return img, mask


class ImageFolder(data.Dataset):

    def __init__(self, datalist, root, img_type, label_type):
        self.ids = datalist
        self.loader = default_loader
        self.root = root
        self.img_type = img_type
        self.label_type = label_type

    def __getitem__(self, index):
        id = self.ids[index]
        copy_index = int(np.random.random() * len(self.ids))
        copy_id = self.ids[copy_index]
        img, mask = self.loader(id, self.root, self.img_type, self.label_type, copy_id)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.ids)