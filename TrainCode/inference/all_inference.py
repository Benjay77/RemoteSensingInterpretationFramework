# coding=utf-8
"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/7/12 14:49
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/7/12 14:49
 *@Description: 推理文件
"""
import argparse
import glob
import os
import random
import sys
import time
import warnings

import numpy as np
from tqdm import tqdm

from common_function import *
from networks import *


def getargs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--image_path', type=str,
                       default='../dataset/imgs')
    parse.add_argument('--save_path', type=str,
                       default='../results/')
    parse.add_argument('--image_size', type=int, default=1024)
    parse.add_argument('--resolution', type=str, default='0.05m')
    parse.add_argument('--interpret_type', type=str, default='build')
    parse.add_argument('--weight_name', type=str, default='202109211150')
    parse.add_argument('--image_channel', type=int, default=3)
    parse.add_argument('--overlay', type=int, default=4096)
    parse.add_argument('--padding_step', type=int, default=2048)
    parse.add_argument('--inference_num', type=int, default=1)
    parse.add_argument('--threshold', type=float, default=0.4)
    parse.add_argument('--small_threshold', type=int, default=1000)
    parse.add_argument('--image_type', type=str, default='.tif')
    parse.add_argument('--label_type', type=str, default='.tif')
    parse.add_argument('--label_flag', type=str, default='_mask')
    parse.add_argument('--arch', '-a', metavar='ARCH', default='UNetR50',
                       help='UNetR50')
    parse.add_argument('--weights', type=str, default='weights', help='path of weights files')
    parse.add_argument('--batchsize_per_card', type=int, default=96)
    parse.add_argument('--use_multiple_GPU', type=bool, default=False)
    parse.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parse.add_argument('--kronecker_r1', nargs='+', type=int,
                       default=[2, 4, 8])
    parse.add_argument('--kronecker_r2', nargs='+', type=int,
                       default=[1, 3, 5])
    parse.add_argument('--log_dir', default='/logs', help="log dir")
    return parse.parse_args()


class InferenceModel:
    def __init__(self, args):
        self.args = args
        if self.args.arch == 'UNetR50':
            self.net = UNetR50()

        # if self.args.use_multiple_GPU:
        #     self.net = torch.nn.DataParallel(self.net.cuda(), device_ids = range(torch.cuda.device_count()))
        #     self.batch_size = torch.cuda.device_count() * self.args.batchsize_per_card
        # else:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     # if device.type == 'cuda':
        #     #     self.net = nn.DataParallel(self.net, device_ids = [0])
        #     # else:
        #     #     device = 'cpu'
        #     self.net = nn.DataParallel(self.net.cuda(device), device_ids = [0])
        #     self.batch_size = self.args.batchsize_per_card
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = device.type
        if device.type == 'cuda':
            if self.args.use_multiple_GPU:
                device_ids = range(torch.cuda.device_count())
                # self.net = nn.parallel.DistributedDataParallel(self.net.cuda(device), device_ids=device_ids)
            else:
                device_ids = [0]
            self.net = nn.DataParallel(self.net.cuda(device), device_ids=device_ids)
            self.batch_size = torch.cuda.device_count() * self.args.batchsize_per_card
        else:
            self.net = nn.DataParallel(self.net)
            self.batch_size = self.args.batchsize_per_card

    def test_one_img_from_path(self, img):
        self.net.eval()
        if self.args.overlay != self.args.image_size:
            return self.crop_prediction(img)
        else:
            if self.batch_size >= 1:
                return self.padding_prediction(img)

    def crop_prediction(self, img):
        img_crop = np.zeros((self.args.overlay, self.args.overlay, self.args.inference_num), dtype=np.float32,
                            order='C')
        img_temp = np.zeros((self.args.overlay, self.args.overlay, self.args.image_channel), dtype=np.uint8,
                            order='C')
        # 记录不为黑色的像素的个数
        ratio = np.ones((self.args.overlay, self.args.overlay), dtype=np.float32, order='C')
        img_ratio = np.zeros((self.args.overlay, self.args.overlay, self.args.inference_num), dtype=np.float32,
                             order='C')

        for num in range(self.args.inference_num):
            # if num == 0:
            #     img_crop[0:self.args.overlay, 0:self.args.overlay, 0] = self.predict_patch(img, self.args.overlay,
            #                                                                                self.args.overlay,
            #                                                                                self.args.image_size)[
            #                                                             0:self.args.overlay,
            #                                                             0:self.args.overlay]
            #     img_ratio[0:self.args.overlay, 0:self.args.overlay, 0] = ratio[0:self.args.overlay, 0:self.args.overlay]
            # else:
            img_crop_up = random.randint(self.args.image_size // 2, self.args.image_size)
            img_crop_left = random.randint(self.args.image_size // 2, self.args.image_size)
            # img_temp[0: img_crop_up, 0:img_crop_left, :] = img[0: img_crop_up, 0:img_crop_left, :]
            # img_crop[0: img_crop_up, 0:img_crop_left, num] = self.predict_patch(
            #     img_temp,
            #     self.args.image_size,
            #     self.args.image_size,
            #     self.args.image_size)[0: img_crop_up, 0:img_crop_left]
            # img_ratio[0: img_crop_up, 0:img_crop_left, num] = ratio[0: img_crop_up, 0:img_crop_left]
            #
            # img_temp[0: self.args.overlay, 0:(self.args.overlay - img_crop_left), :] = img[
            #                                                                            0: self.args.overlay,
            #                                                                            img_crop_left:self.args.overlay,
            #                                                                            :]
            # img_crop[0: self.args.overlay, img_crop_left:self.args.overlay, num] = self.predict_patch(
            #     img_temp,
            #     self.args.overlay,
            #     self.args.overlay,
            #     self.args.image_size)[0: self.args.overlay, 0:(self.args.overlay - img_crop_left)]
            # img_ratio[0: self.args.overlay, img_crop_left:self.args.overlay, num] = ratio[
            #                                                                         0: self.args.overlay,
            #                                                                         img_crop_left:self.args.overlay]
            #
            # img_temp[0:(self.args.overlay - img_crop_up), 0: self.args.overlay, :] = img[
            #                                                                          img_crop_up:self.args.overlay,
            #                                                                          0:self.args.overlay,
            #                                                                          :]
            # img_crop[img_crop_up:self.args.overlay, 0:self.args.overlay, num] = self.predict_patch(
            #     img_temp,
            #     self.args.overlay,
            #     self.args.overlay,
            #     self.args.image_size)[0:(self.args.overlay - img_crop_up), 0:self.args.overlay]
            # img_ratio[img_crop_up:self.args.overlay, 0:self.args.overlay, num] = ratio[
            #                                                                      img_crop_up:self.args.overlay,
            #                                                                      0:self.args.overlay]
            img_temp_up = random.randint(img_crop_up // 2, img_crop_up)
            img_temp_left = random.randint(img_crop_left // 2, img_crop_left)
            img_temp[img_temp_up:(self.args.overlay - img_crop_up + img_temp_up),
            img_temp_left:(self.args.overlay -
                           img_crop_left + img_temp_left),
            :] = img[img_crop_up:self.args.overlay, img_crop_left:self.args.overlay, :]
            # img_temp[img_crop_up:self.args.overlay, img_crop_left:self.args.overlay,
            # :] = img[img_crop_up:self.args.overlay, img_crop_left:self.args.overlay, :]
            img_crop[img_crop_up:self.args.overlay, img_crop_left:self.args.overlay, num] = self.predict_patch(
                img_temp,
                self.args.overlay,
                self.args.overlay,
                self.args.image_size)[img_temp_up:(self.args.overlay - img_crop_up + img_temp_up),
                                                                                            img_temp_left:(
                                                                                                    self.args.overlay -
                                                                                                    img_crop_left +
                                                                                                    img_temp_left)]
            img_ratio[img_crop_up:self.args.overlay, img_crop_left:self.args.overlay, num] = ratio[
                                                                                             img_crop_up:self.args.overlay,
                                                                                             img_crop_left:self.args.overlay]

        img_crop_sum = img_crop.sum(axis=2)
        img_ratio_sum = img_ratio.sum(axis=2)
        prediction = img_crop_sum / img_ratio_sum

        # 预测
        prediction = np.where(prediction >= self.args.threshold, 1, 0)
        prediction = np.array(prediction, dtype='uint8')
        # 剔除噪音
        prediction = post_process(prediction, args.small_threshold, 1, self.args.interpret_type, self.args.threshold)
        del img_crop, img_temp, ratio, img_ratio
        return prediction

    def predict_patch(self, img, img_row, img_col, step):
        patch_prediction = np.zeros((img_row, img_col), dtype=np.float32, order='C')
        # 分块读取影像，分块预测
        # 逐列提取影像
        for c in range(0, img_col, step):
            # 截取部分影像
            img_part = img[0: img_row, c: c + step, :]
            img_part_sum = img_part.sum(axis=2)
            if img_part_sum.max() == 0:
                patch_prediction[0: img_row, c: c + step] = 0
            else:
                if self.args.interpret_type != 'water':  # and self.args.interpret_type != 'build'
                    img = limit_histogram_equalization(img)
                # 分解出三个波段，并且加一个轴
                img_part_r = np.expand_dims(img_part[:, :, 0], 0)
                img_part_g = np.expand_dims(img_part[:, :, 1], 0)
                img_part_b = np.expand_dims(img_part[:, :, 2], 0)
                # 整成多个波段
                img_part_r_16 = img_part_r.reshape((img_col // step, step, step), order='C')
                img_part_g_16 = img_part_g.reshape((img_col // step, step, step), order='C')
                img_part_b_16 = img_part_b.reshape((img_col // step, step, step), order='C')
                # 合并各个波段
                img_part_r_16 = np.expand_dims(img_part_r_16, 1)
                img_part_g_16 = np.expand_dims(img_part_g_16, 1)
                img_part_b_16 = np.expand_dims(img_part_b_16, 1)
                img_part_rgb = np.concatenate((img_part_r_16, img_part_g_16, img_part_b_16), axis=1)
                img_part_rgb = img_part_rgb.astype(np.float32) / 255.0 * 3.2 - 1.6
                if self.device_type == 'cuda':
                    img_part_rgb = V(torch.Tensor(img_part_rgb).cuda())
                else:
                    img_part_rgb = V(torch.Tensor(img_part_rgb))

                # 开始预测
                with torch.no_grad():
                    temp = self.net.forward(img_part_rgb).reshape(1, img_row, step).squeeze().cpu().data.numpy()
                    # temp = temp.data.cpu().numpy()
                    # temp = np.squeeze(temp, axis=1)

                    # temp = temp[:, :, :].reshape((1, img_row, step), order='C')
                    # temp = np.squeeze(temp, axis=None)
                    patch_prediction[0: img_row, c: c + step] = temp

        return patch_prediction

    def padding_prediction(self, img):
        if self.args.interpret_type != 'water':  # and self.args.interpret_type != 'build':
            img = limit_histogram_equalization(img)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])

        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])

        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose((0, 3, 1, 2))
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img6 = img4.transpose((0, 3, 1, 2))
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        if self.device_type == 'cuda':
            img5 = V(torch.Tensor(img5).cuda())
            img6 = V(torch.Tensor(img6).cuda())
        else:
            img5 = V(torch.Tensor(img5))
            img6 = V(torch.Tensor(img6))

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = (maska + maskb[:, :, ::-1])
        mask2 = (mask1[:2] + mask1[2:, ::-1])
        mask3 = (mask2[0] + np.rot90(mask2[1])[::-1, ::-1]) / 8
        mask3 = np.where(mask3 > self.args.threshold, 1, 0)
        # mask3 = dense_crf(img, mask3)
        mask3 = np.array(mask3, dtype='uint8')
        mask3 = post_process(mask3, self.args.small_threshold, 1, self.args.interpret_type, self.args.threshold, img)

        return mask3

    def load(self, model_path):
        if self.device_type == 'cuda':
            self.net.load_state_dict(torch.load(model_path))
        else:
            self.net.load_state_dict(torch.load(model_path, 'cpu'))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = getargs()
    path = os.path.abspath(sys.argv[0])
    (file_path, temp_filename) = os.path.split(path)

    ##########################
    # img_dir = sys.argv[1]
    # save_dir = sys.argv[2]
    # img_path = img_dir.encode("utf-8").decode("utf-8")
    # save_dir = save_dir.encode("utf-8").decode("utf-8")
    print('\n>>>>>>>>>>>>>>>>>>>>>>>> begin load model >>>>>>>>>>>>>>>>>>>>>>>>>>>')
    time_model_begin = time.time()
    solver = InferenceModel(args)
    model_path = os.path.join(file_path, args.weights, args.resolution, args.interpret_type, args.interpret_type +
                              args.weight_name,
                              args.interpret_type + args.weight_name + '_best.pth')
    if not os.path.exists(model_path):
        print("**************** model not exists *********************************")
        sys.exit()
    solver.load(model_path)
    print(f'model load cost{time.time() - time_model_begin}s')
    print(">>>>>>>>>>>>>>>>>>>>> model load succeed >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    img_path = args.image_path.encode("utf-8").decode("utf-8")
    save_path = os.path.join(args.save_path, args.resolution, args.interpret_type,
                             args.interpret_type + args.weight_name + '_' + str(args.overlay) + '_' + str(args.inference_num) +
                             '_' + str(args.threshold) + '_' + str(args.small_threshold)).encode(
        "utf-8").decode("utf-8")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    files_path = glob.glob(os.path.join(img_path, '*' + args.image_type))
    time_detect_begin = time.time()
    loop = tqdm(files_path)
    for name in loop:
        if args.image_type == '.tif' or args.image_type == '.tiff':
            image, img_info = read_tiff(name)
            h, w, channel = img_info['row'], img_info['col'], img_info['bands']
        else:
            image = cv2.imdecode(np.fromfile(name, dtype=np.uint8), 1)
            h, w, channel = image.shape[0], image.shape[1], image.shape[2]
        if h < args.overlay or w < args.overlay:
            solver.args.overlay = args.overlay = args.image_size
            solver.args.padding_step = args.padding_step = args.image_size // 2
        row_end = col_end = False
        if h % args.padding_step == 0:
            padding_h = h
        else:
            padding_h = (h // args.padding_step + 1) * args.padding_step
        if w % args.padding_step == 0:
            padding_w = w
        else:
            padding_w = (w // args.padding_step + 1) * args.padding_step
        # padding_img = np.zeros((padding_h, padding_w, args.image_channel), dtype = np.uint8, order = 'C')
        # padding_img[0:h, 0:w, :] = img_array[:, :, :]
        if args.padding_step > args.image_size:
            padding_border_left = random.randint(args.image_size, args.padding_step)
            padding_border_top = random.randint(args.image_size, args.padding_step)
        else:
            padding_border_left = random.randint(args.padding_step // 2, args.padding_step)
            padding_border_top = random.randint(args.padding_step // 2, args.padding_step)
        # padding_border_left = padding_border_top = args.padding_step // 2
        padding_border_right = args.padding_step - padding_border_left
        padding_border_bottom = args.padding_step - padding_border_top
        # padding_img = cv2.copyMakeBorder(padding_img, padding_border_top, padding_border_bottom,
        #                                  padding_border_left,
        #                                  padding_border_right, cv2.BORDER_CONSTANT, 0)
        # padding_h, padding_w = padding_img.shape[0], padding_img.shape[1]
        padding_h = padding_h + args.padding_step
        padding_w = padding_w + args.padding_step
        pred_img = np.zeros((args.padding_step, args.padding_step), dtype=np.uint8)
        # pred_offset_img = np.zeros((args.padding_step, args.padding_step), dtype=np.uint8)
        crop = np.zeros((args.overlay, args.overlay, 3), dtype=np.uint8)
        # crop_offset = np.zeros((args.overlay, args.overlay, 3), dtype=np.uint8)
        prediction_buffer = np.zeros((padding_h - args.padding_step, padding_w - args.padding_step), dtype=np.uint8, order='C')
        # offset_h = offset_w = 0
        # img_up_offset = img_down_offset = img_left_offset = img_right_offset = 0
        for i in range(padding_h // args.padding_step):
            if row_end and col_end:
                break
            else:
                col_end = False
            up = i * args.padding_step
            down = up + args.overlay
            if down == padding_h:
                row_end = True
            else:
                row_end = False
            img_down = down - padding_border_top
            if down - padding_border_top >= h:
                img_down = h
            if i == 0:
                img_up = 0
                crop_up = padding_border_top
            else:
                img_up = up - padding_border_top
                if img_up >= h:
                    break
                crop_up = 0
                # crop_down = img_down-img_up
            crop_down = args.overlay
            if img_down == h:
                crop_down = img_down - img_up
            # if i > 0:
            #     offset_h = random.randint((args.padding_step - padding_border_top) // 4,
            #                               (args.padding_step - padding_border_top) // 2)
            #     crop_up_offset = 0
            # else:
            #     offset_h = 0
            #     crop_up_offset = padding_border_top
            # img_up_offset = img_up - offset_h
            # img_down_offset = img_down - offset_h
            # crop_down_offset = args.overlay
            # if img_down == h:
            #     crop_down_offset = img_down_offset - img_up_offset
            for j in range(padding_w // args.padding_step):
                if col_end:
                    break
                cuda_mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
                loop.set_description(f'CUDA {cuda_mem:.3g}GB')
                loop.set_postfix(image=os.path.basename(name), row=i + 1, rows=padding_h // args.padding_step - 1,
                                 col=j + 1, cols=padding_w // args.padding_step - 1)
                left = j * args.padding_step
                right = left + args.overlay
                if right == padding_w:
                    col_end = True
                else:
                    col_end = False
                img_right = right - padding_border_left
                if right - padding_border_left >= w:
                    img_right = w
                if j == 0:
                    img_left = 0
                    crop_left = padding_border_left
                else:
                    img_left = left - padding_border_left
                    if img_left >= w:
                        break
                    crop_left = 0
                    # crop_right = img_right-img_left
                crop_right = args.overlay
                if img_right == w:
                    crop_right = img_right - img_left
                if args.image_type == '.tif' or args.image_type == '.tiff':
                    img_array = image.ReadAsArray(img_left, img_up, img_right - img_left, img_down - img_up).astype(
                        np.uint8)[0:3,
                                :, :].transpose(1, 2, 0)
                else:
                    img_array = image[img_up: img_down, img_left: img_right]
                crop[crop_up:crop_down, crop_left:crop_right] = img_array
                crop_sum = crop.sum(axis=2)
                crop_sum[crop_sum > 0] = 1
                if crop.max() == 0:
                    pred_img[:, :] = 0
                else:
                    temp_prediction = solver.test_one_img_from_path(crop) * crop_sum.astype(
                        np.uint8)
                    pred_img = temp_prediction[
                               padding_border_top:padding_border_top + args.padding_step,
                               padding_border_left:padding_border_left + args.padding_step]
                    pred_img = pred_img.astype(np.uint8)

                    prediction_buffer[i * args.padding_step:i * args.padding_step + args.padding_step,
                    j * args.padding_step:j * args.padding_step + args.padding_step] = pred_img[:, :]

                # if j > 0:
                #     offset_w = random.randint((args.padding_step - padding_border_left) // 4,
                #                               (args.padding_step - padding_border_left) // 2)
                #     crop_left_offset = 0
                # else:
                #     offset_w = 0
                #     crop_left_offset = padding_border_left
                # img_left_offset = img_left - offset_w
                # img_right_offset = img_right - offset_w
                # crop_right_offset = args.overlay
                # if img_right == w:
                #     crop_right_offset = img_right_offset - img_left_offset
                # if (i > 0 or j > 0) and img_down_offset <= h and img_right_offset <= w:
                #     if args.image_type == '.tif' or args.image_type == '.tiff':
                #         img_array_offset = image.ReadAsArray(img_left_offset, img_up_offset,
                #                                              img_right_offset - img_left_offset,
                #                                              img_down_offset - img_up_offset).astype(np.uint8)[0:3, :,
                #                            :].transpose(
                #             1,
                #             2, 0)
                #     else:
                #         img_array_offset = image[img_up_offset: img_down_offset, img_left_offset: img_right_offset]
                #     crop_offset[crop_up_offset:crop_down_offset, crop_left_offset:crop_right_offset] = img_array_offset
                #     crop_offset_sum = crop_offset.sum(axis=2)
                #     crop_offset_sum[crop_offset_sum > 0] = 1
                #     if crop_offset.max() == 0:
                #         pred_offset_img[:, :] = 0
                #     else:
                #         temp_prediction = solver.test_one_img_from_path(crop_offset) * crop_offset_sum.astype(
                #             np.uint8)
                #         pred_offset_img = temp_prediction[
                #                           padding_border_top:padding_border_top +
                #                                              args.padding_step,
                #                           padding_border_left:padding_border_left +
                #                                               args.padding_step]
                #         pred_offset_img = pred_offset_img.astype(np.uint8)
                #
                #         prediction_buffer[
                #         i * args.padding_step - offset_h:i * args.padding_step - offset_h + args.padding_step,
                #         j * args.padding_step - offset_w:j * args.padding_step - offset_w + args.padding_step] = \
                #             pred_offset_img[:, :]
        mask_img = prediction_buffer[0:h, 0:w]
        mask_img = np.array(mask_img, dtype='uint8')
        # mask_img = dense_crf(image, mask_img)
        image_array = image.ReadAsArray().astype(np.uint8)[0:3, :, :].transpose(1, 2, 0)
        mask_img = post_process(mask_img, args.small_threshold, 1, args.interpret_type, args.threshold, image_array)
        mask_img = mask_img.astype(np.uint8) * 255
        del prediction_buffer, pred_img, crop  # , padding_img, pred_offset_img,

        img_name = (os.path.basename(name)).split(args.image_type)[0]
        raster_path = os.path.join(save_path, img_name + args.label_flag + args.label_type)
        if args.image_type == '.tif' or args.image_type == '.tiff':
            write_tiff(im_data=mask_img, im_geotrans=img_info['geotrans'], im_proj=img_info['geoproj'],
                       path_out=raster_path)
            # C = ResetCoord(name, raster_path)
            # C.assign_spatial_reference_by_file()  # 添加空间参考系
            shp_path = os.path.join(save_path, img_name + '.shp')
            ShapeFile(raster_path, shp_path).create_shapefile()
        else:
            cv2.imencode(args.image_type, mask_img)[1].tofile(raster_path)
        # cv2.imwrite(raster_path, mask_img)  # 保存预测结果

    print(f'Detect cost{time.time() - time_detect_begin}s')
