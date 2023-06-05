"""
Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]
Dataset processing.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/GFM
Paper link (Arxiv): https://arxiv.org/abs/2010.16188

"""
import kornia
import torch
import cv2
import os
import random
import numpy as np
from PIL import Image
from datasets.Base_dataset import Base_Dataset, TRAIN_TYPE, VAL_TYPE, SHOW_TYPE
import torchvision.transforms as T
from skimage.transform import resize
########## Parameters for training
from utils.DIMUtil import getInstanceMap, generateRandomPriorDIM
from utils.util import generateRandomPrior, scale_img, padding_to_square, get_trimap, show_tensor  # , scale_img_p3m

import matplotlib.pyplot as plt


########## Parameters for testing
# MAX_SIZE_H = 1600
# MAX_SIZE_W = 1600
# SHORTER_PATH_LIMITATION = 1080


#########################
## Data transformer
#########################


class MattingTransform(object):
    def __init__(self, out_size=512, crop_size=[512, 768, 1024]):
        super(MattingTransform, self).__init__()
        self.out_size = out_size
        self.crop_size = crop_size

    def __call__(self, *argv):
        ori = argv[0]
        h, w, c = ori.shape
        rand_ind = random.randint(0, len(self.crop_size) - 1)
        crop_size = self.crop_size[rand_ind] if self.crop_size[rand_ind] < min(h, w) else 320
        resize_size = self.out_size
        ### generate crop centered in transition area randomly
        trimap = argv[1]
        trimap_crop = trimap[:h - crop_size, :w - crop_size]
        target = np.where(trimap_crop == 128) if random.random() < 0.5 else np.where(trimap_crop > -100)
        if len(target[0]) == 0:
            target = np.where(trimap_crop > -100)

        rand_ind = np.random.randint(len(target[0]), size=1)[0]
        cropx, cropy = target[1][rand_ind], target[0][rand_ind]
        # # flip the samples randomly
        flip_flag = True if random.random() < 0.5 else False
        # generate samples (crop, flip, resize)
        argv_transform = []
        index = 0
        for item in argv:
            item = item[cropy:cropy + crop_size, cropx:cropx + crop_size]
            if flip_flag:
                item = cv2.flip(item, 1)
            if index == 1:
                item = cv2.resize(item, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
            item = cv2.resize(item, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
            argv_transform.append(item)
            index += 1

        return argv_transform


def process_fgbg(ori, mask, is_fg, fgbg_path=None):
    if fgbg_path is not None:
        img = np.array(Image.open(fgbg_path))
    else:
        mask_3 = (mask / 255.0)[:, :, np.newaxis].astype(np.float32)

        if is_fg:
            mask_3[mask_3 != 0] = 1
            img = ori * mask_3
        else:
            mask_3[mask_3 != 1] = 0
            img = ori * (1 - mask_3)
    return img


def resize_img(ori, img):
    img = cv2.resize(img, ori.shape) * 255.0
    return img


def add_guassian_noise(img, fg, bg):
    row, col, ch = img.shape
    mean = 0
    sigma = 10
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_img = np.uint8(img + gauss)
    noisy_fg = np.uint8(fg + gauss)
    noisy_bg = np.uint8(bg + gauss)
    return noisy_img, noisy_fg, noisy_bg


def generate_composite_rssn(fg, bg, mask, fg_denoise=None, bg_denoise=None):
    ## resize bg accordingly
    h, w, c = fg.shape
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = mask / 255.
    bg = resize_img(fg, bg)
    ## use denoise fg/bg randomly
    if fg_denoise is not None and random.random() < 0.5:
        fg = fg_denoise
        bg = resize_img(fg, bg_denoise)
    ## reduce sharpness discrepancy
    if random.random() < 0.5:
        rand_kernel = random.choice([20, 30, 40, 50, 60])
        bg = cv2.blur(bg, (rand_kernel, rand_kernel))
    composite = alpha * fg + (1 - alpha) * bg
    composite = composite.astype(np.uint8)
    ## reduce noise discrepancy
    if random.random() < 0.5:
        composite, fg, bg = add_guassian_noise(composite, fg, bg)
    return composite, fg, bg


def generate_composite_coco(fg, bg, mask):
    h, w, c = fg.shape
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = mask / 255.
    bg = resize_img(fg, bg)
    composite = alpha * fg + (1 - alpha) * bg
    composite = composite.astype(np.uint8)
    return composite, fg, bg


def gen_trimap_with_dilate(alpha, kernel_size):
    h, w = alpha.shape
    alpha = cv2.resize(alpha, (512, 512), cv2.INTER_LINEAR)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)
    erode = cv2.erode(fg, kernel, iterations=1)
    trimap = erode * 255 + (dilate - erode) * 128
    trimap = cv2.resize(trimap, (w, h), cv2.INTER_NEAREST)
    return trimap.astype(np.uint8)


def save_test_result(save_dir, predict):
    predict = (predict * 255).astype(np.uint8)
    cv2.imwrite(save_dir, predict)


#########################
## Data Loader
#########################
class P3MP_Dataset(Base_Dataset):
    def __init__(self, args, mode='train'):
        super().__init__(args, mode)

        self.out_size = args.crop_size
        self.val_size = args.val_size
        self.mode = mode

        if mode == 'train':
            self.transform = MattingTransform(out_size=self.out_size, crop_size=[512, 768, 1024])
            self.imPath = args.im_path
            self.gtPath = args.gt_path
        elif mode == 'val':
            self.prior_dict = {}
            self.transform = T.Compose([
                T.ToTensor(),
            ])
            self.imPath = args.val_img_path
            self.gtPath = args.val_gt_path
        elif mode == 'show':
            self.prior_dict = {}
            self.transform = T.Compose([
                T.ToTensor(),
            ])
            self.imPath = args.show_img_path
            self.gtPath = args.show_gt_path

        # self.RSSN_DENOISE = args.rssn_denoise

        self.merge_files = os.listdir(self.imPath)
        self.gt_files = os.listdir(self.gtPath)

        if args.fg_path is not None and args.fg_path != '':
            self.fgPath = args.fg_path
            self.fg_files = os.listdir(args.fg_path)
            self.fg_files.sort()
        else:
            self.fgPath = None

        if args.bg_path is not None and args.bg_path != '':
            self.bgPath = args.bg_path
            self.bg_files = os.listdir(args.bg_path)
            self.bg_files.sort()
        else:
            self.bgPath = None

        if args.fgPath_denoise is not None and args.bgPath_denoise is not None \
                and args.fgPath_denoise != '' and args.bgPath_denoise != '':
            self.bgPath_denoise = args.bgPath_denoise
            self.fgPath_denoise = args.fgPath_denoise
            self.fg_files_denoise = os.listdir(self.fgPath_denoise)
            self.bg_files_denoise = os.listdir(self.bgPath_denoise)
            self.fg_files_denoise.sort()
            self.bg_files_denoise.sort()
        else:
            self.bgPath_denoise = None
            self.fgPath_denoise = None
        self.merge_files.sort()
        self.gt_files.sort()
        self.trimap_path = args.trimap_path
        self.use_user_map = args.use_user_map

        print('{} numbers: {}'.format(mode, len(self.merge_files)))

    def get_train_data(self, item: int) -> TRAIN_TYPE:
        fg_path = None
        bg_path = None
        im_index = item % len(self.merge_files)
        im_name = self.merge_files[im_index]
        label_name = self.gt_files[im_index]
        assert label_name.split('.')[0] == im_name.split('.')[0], 'name is not match'
        merge_im = np.array(Image.open(self.imPath + im_name))
        label_im = np.array(Image.open(self.gtPath + label_name))

        # h, w, c = merge_im.shape
        # new_h = min(self.val_size, h)
        # new_w = min(self.val_size, w)
        # merge_im = cv2.resize(merge_im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # label_im = cv2.resize(label_im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if len(label_im.shape) == 2:
            label_im = label_im
        else:
            label_im = label_im[:, :, 0]

        # merge_im, label_im = self._composite_fg(merge_im, label_im, item)

        if self.fgPath is not None:
            fg_path = self.fgPath + self.fg_files[item]
        if self.bgPath is not None:
            bg_path = self.bgPath + self.bg_files[item]
        # fg = np.array(Image.open(fg_path))
        # bg = np.array(Image.open(bg_path))

        fg = process_fgbg(merge_im, label_im, True, fg_path)
        bg = process_fgbg(merge_im, label_im, False, bg_path)

        kernel_size_tt = random.randint(15, 30)
        trimap = gen_trimap_with_dilate(label_im, kernel_size_tt)

        # Data transformation to generate samples
        # crop/flip/resize
        argv = self.transform(merge_im, trimap, label_im, fg, bg)
        argv_transform = []
        for items in argv:
            if items.ndim < 3:
                items = torch.from_numpy(items.astype(np.float32)[np.newaxis, :, :]) / 255.0
            else:
                items = torch.from_numpy(items.astype(np.float32)).permute(2, 0, 1) / 255.0
            argv_transform.append(items)

        [ori, trimap, mask, fg, bg] = argv_transform
        trimap[(trimap != 0) * (trimap != 1)] = 0.5

        # prior = generateRandomPrior(argv[2], size=31)
        # prior_trimap = prior.copy()
        # prior_trimap[prior_trimap == -1] = 1

        if self.use_user_map:
            prior = torch.from_numpy(generateRandomPriorDIM(mask[0].cpu().numpy(), r=5, mode='rect')).unsqueeze(0)
        else:
            prior = mask

        # shadow
        # if random.random() < 0.5:
        #     aug_shadow = mask.mul(max(0.05, random.random()))
        #     aug_shadow = T.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.3), scale=(0.95, 1.1), shear=(-5, 5))(
        #         aug_shadow)
        #     aug_shadow = kornia.filters.box_blur(aug_shadow.unsqueeze(0), (random.choice(range(20, 40)),) * 2)
        #     bg = (bg - aug_shadow[0]).clamp_(0, 1)
        #     # show_tensor(ori.permute([1, 2, 0]))
        #     # show_tensor(aug_shadow[0][0], mode='jet')
        #     ori = fg * mask + (1 - mask) * bg
        #     # show_tensor(bg.permute([1, 2, 0]))
        #     # show_tensor(ori.permute([1, 2, 0]))

        # RGB BGR
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        ori = normalize(ori)
        fg = normalize(fg)
        bg = normalize(bg)

        return ori, mask, trimap, fg, bg, prior, item

    def get_val_data(self, item: int) -> VAL_TYPE:
        fg_index = item % len(self.merge_files)
        fg_name = self.merge_files[fg_index]
        label_name = self.gt_files[fg_index]

        assert label_name.split('.')[0] == fg_name.split('.')[0], 'name is not match'

        fg_im = np.array(Image.open(self.imPath + fg_name).convert('RGB'))
        label_im = np.array(Image.open(self.gtPath + label_name))
        # trimap_im = np.array(Image.open(self.trimap_path + trimap_name))
        if len(label_im.shape) == 2:
            label_im = label_im
        # elif label_im.shape[2] == 4:
        #     label_im = label_im[..., -1]
        else:
            label_im = label_im[:, :, 0]

        fg_size = fg_im.shape
        h, w = fg_size[0], fg_size[1]
        # h,w = 1600, 1080
        resized_h = int(h / 2)
        resized_w = int(w / 2)
        # resized_h = int(h)
        # resized_w = int(w)
        # new_h = min(self.val_size, resized_h - (resized_h % 32))
        # new_w = min(self.val_size, resized_w - (resized_w % 32))
        new_h = resized_h - (resized_h % 32)
        new_w = resized_w - (resized_w % 32)
        # resize_h = int(h / 2)
        # resize_w = int(w / 2)
        # new_h = min(self.val_size, resize_h - (resize_h % 32))
        # new_w = min(self.val_size, resize_w - (resize_w % 32))
        # fg_im = scale_img_p3m(fg_im, mode='long', skip_small=True, size=self.val_size)
        # label_im = scale_img_p3m(label_im, mode='long', skip_small=True, size=self.val_size)
        fg_im = resize(fg_im, (new_h, new_w)) * 255.0
        # label_im = resize(label_im, (new_h, new_w))

        # fg_im = padding_to_square(fg_im)
        # label_im = padding_to_square(label_im)
        # fg_im = fg_im.astype(np.float32)[:, :, :].transpose(2, 0, 1)
        fg_im = torch.from_numpy(fg_im.astype(np.float32)[:, :, :]).permute(2, 0, 1) / 255.0
        merge_img, label_alpha = fg_im, label_im

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        # normalize = T.Normalize(mean=[0.406, 0.456, 0.485],
        #                         std=[0.225, 0.224, 0.229])

        # merge_img = self.transform(merge_img)
        merge_img = normalize(merge_img)
        merge_gt = self.transform(np.array(label_alpha, dtype='uint8'))

        trimap = torch.from_numpy(get_trimap(merge_gt[0].cpu().numpy()))
        # trimap_g = torch.from_numpy(trimap)
        # trimap_t = trimap.data.numpy()

        if label_name in self.prior_dict.keys():
            prior = self.prior_dict[label_name]
        else:
            prior = generateRandomPriorDIM(merge_gt[0].cpu().numpy(), r=5, mode='rect', val=False)
            self.prior_dict[label_name] = prior

        return merge_img, merge_gt, trimap, prior, item

    def get_show_data(self, item: int) -> SHOW_TYPE:
        merge_img, merge_gt, trimap, item = self.get_val_data(item)
        return merge_img, merge_gt, item

    def _composite_fg(self, fg, alpha, idx):
        if np.random.rand() < 0.5:
            idx2 = np.random.randint(len(self.merge_files)) + idx
            idx2 = idx2 % len(self.merge_files)
            im_name = self.merge_files[idx2]
            label_name = self.gt_files[idx2]
            fg2 = np.array(Image.open(self.imPath + im_name))
            alpha2 = np.array(Image.open(self.gtPath + label_name).convert('L'))
            h, w = alpha.shape
            fg2 = cv2.resize(fg2, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha2 = cv2.resize(alpha2, (w, h), interpolation=cv2.INTER_LINEAR)

            alpha = alpha.astype(float) / 255.0
            alpha2 = alpha2.astype(float) / 255.0

            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            if np.any(alpha_tmp < 1):
                fg = fg.astype(float) * alpha[:, :, None] + fg2.astype(float) * (1 - alpha[:, :, None])
                # The overlap of two 50% transparency should be 25%
                alpha = alpha_tmp
                fg = fg.astype(np.uint8)
            alpha = np.array(alpha * 255, dtype='uint8')

            # plt.subplot(1,3,1)
            # plt.imshow(fg)
            # plt.subplot(1, 3, 2)
            # plt.imshow(fg2)
            # plt.subplot(1, 3, 3)
            # plt.imshow(alpha)
            # plt.show()
            # print()
        return fg, alpha

    def __len__(self):
        return len(self.merge_files)

# def inference_img(img, maxs):
#     h, w, c = img.shape
#     new_h = min(max, h - (h % 32))
#     new_w = min(max, w - (w % 32))
#
#     resize_h = int(h / 2)
#     resize_w = int(w / 2)
#     new_h = min(max, resize_h - (resize_h % 32))
#     new_w = min(max, resize_w - (resize_w % 32))
#     scale_img = resize(img, (new_h, new_w)) * 255.0
#
#     return scale_img
#
#
# class P3M_Dataset(Base_Dataset):
#     def __init__(self, args, mode='train'):
#         super().__init__(args, mode)
#
#         self.out_size = args.crop_size
#         self.val_size = args.val_size
#         self.mode = mode
#
#         if mode == 'train':
#             self.transform = MattingTransform(out_size=self.out_size, crop_size=[512, 768, 1024])
#             self.imPath = args.im_path
#             self.gtPath = args.gt_path
#         elif mode == 'val':
#             self.transform = T.Compose([
#                 T.ToTensor(),
#             ])
#             self.imPath = args.val_img_path
#             self.gtPath = args.val_gt_path
#         elif mode == 'show':
#             self.transform = T.Compose([
#                 T.ToTensor(),
#             ])
#             self.imPath = args.show_img_path
#             self.gtPath = args.show_gt_path
#
#         self.RSSN_DENOISE = args.rssn_denoise
#
#         self.merge_files = os.listdir(self.imPath)
#         self.gt_files = os.listdir(self.gtPath)
#
#         if args.fg_path is not None and args.fg_path != '':
#             self.fgPath = args.fg_path
#             self.fg_files = os.listdir(args.fg_path)
#             self.fg_files.sort()
#         else:
#             self.fgPath = None
#
#         if args.bg_path is not None and args.bg_path != '':
#             self.bgPath = args.bg_path
#             self.bg_files = os.listdir(args.bg_path)
#             self.bg_files.sort()
#         else:
#             self.bgPath = None
#
#         if args.fgPath_denoise is not None and args.bgPath_denoise is not None \
#                 and args.fgPath_denoise != '' and args.bgPath_denoise != '':
#             self.bgPath_denoise = args.bgPath_denoise
#             self.fgPath_denoise = args.fgPath_denoise
#             self.fg_files_denoise = os.listdir(self.fgPath_denoise)
#             self.bg_files_denoise = os.listdir(self.bgPath_denoise)
#             self.fg_files_denoise.sort()
#             self.bg_files_denoise.sort()
#         else:
#             self.bgPath_denoise = None
#             self.fgPath_denoise = None
#         self.merge_files.sort()
#         self.gt_files.sort()
#         print('{} numbers: {}'.format(mode, len(self.merge_files)))
#
#     def get_train_data(self, item: int) -> TRAIN_TYPE:
#         im_index = item % len(self.merge_files)
#         im_name = self.merge_files[im_index]
#         label_name = self.gt_files[im_index]
#         assert label_name.split('.')[0] == im_name.split('.')[0], 'name is not match'
#         merge_im = Image.open(self.imPath + im_name)
#         label_im = Image.open(self.gtPath + label_name)
#
#         if len(label_im.shape) == 2:
#             label_im = label_im
#         elif label_im.shape[2] == 4:
#             label_im = label_im[..., -1]
#         else:
#             label_im = label_im[:, :, 0]
#
#         if self.fgPath is not None:
#             fg_path = self.fgPath + self.fg_files[item]
#             fg = np.array(Image.open(fg_path))
#
#         else:
#             fg_path = None
#         if self.bgPath is not None:
#             bg_path = self.bgPath + self.bg_files[item]
#             bg = np.array(Image.open(bg_path))
#         else:
#             bg_path = None
#         # fg = np.array(Image.open(fg_path))
#         # bg = np.array(Image.open(bg_path))
#
#         # fg = process_fgbg(merge_im, label_im, True, fg_path)
#         # bg = process_fgbg(merge_im, label_im, False, bg_path)
#
#         # image = merge_im / 255.0
#         # alpha = label_im / 255.0
#         # fg, bg = solve_foreground_background(image, alpha)
#
#
#         if self.fgPath_denoise is not None and self.bgPath_denoise is not None:
#             fg_denoise = process_fgbg(merge_im, label_im, True, self.fg_files_denoise) if self.RSSN_DENOISE else None
#             bg_denoise = process_fgbg(merge_im, label_im, False, self.bg_files_denoise) if self.RSSN_DENOISE else None
#             merge_im, fg, bg = generate_composite_rssn(fg, bg, label_im, fg_denoise, bg_denoise)
#
#         # ori, fg, bg = generate_composite_coco(fg, bg, label_im)
#
#         # Generate trimap/dilation/erosion online
#         # kernel_size_tt = 25
#         kernel_size_tt = random.randint(15, 30)
#         trimap = gen_trimap_with_dilate(label_im, kernel_size_tt)
#
#         # Data transformation to generate samples
#         # crop/flip/resize
#         argv = self.transform(merge_im, trimap, label_im, fg, bg)
#         argv_transform = []
#         for items in argv:
#             if items.ndim < 3:
#                 items = torch.from_numpy(items.astype(np.float32)[np.newaxis, :, :]) / 255.0
#             else:
#                 items = torch.from_numpy(items.astype(np.float32)).permute(2, 0, 1) / 255.0
#             argv_transform.append(items)
#
#         [ori, trimap, mask, fg, bg] = argv_transform
#         trimap[(trimap != 0) * (trimap != 1)] = 0.5
#         prior = generateRandomPrior(argv[2], size=31)
#         prior_trimap = prior.copy()
#         prior_trimap[prior_trimap == -1] = 1
#         # RGB BGR
#         normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
#
#         ori = normalize(ori)
#         fg = normalize(fg)
#         bg = normalize(bg)
#
#
#         return ori, mask, trimap, fg, bg, torch.from_numpy(prior).unsqueeze(0), torch.from_numpy(
#             prior_trimap).unsqueeze(0), item
#
#     def get_val_data(self, item: int) -> VAL_TYPE:
#         fg_index = item % len(self.merge_files)
#         fg_name = self.merge_files[fg_index]
#         label_name = self.gt_files[fg_index]
#         assert label_name.split('.')[0] == fg_name.split('.')[0], 'name is not match'
#         fg_im = np.array(Image.open(self.imPath + fg_name))
#         label_im = np.array(Image.open(self.gtPath + label_name))
#         if len(label_im.shape) == 2:
#             label_im = label_im
#         elif label_im.shape[2] == 4:
#             label_im = label_im[..., -1]
#         else:
#             label_im = label_im[:, :, 0]
#
#         fg_im = inference_img(fg_im, maxs = self.val_size)
#         label_im = inference_img(label_im, maxs = self.val_size)
#         tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1)
#         # fg_im = scale_img_p3m(fg_im, mode='long', skip_small=True, size=self.val_size)
#         # label_im = scale_img_p3m(label_im, mode='long', skip_small=True, size=self.val_size)
#         # fg_im = padding_to_square(fg_im)
#         # label_im = padding_to_square(label_im)
#         merge_img, label_alpha = fg_im, label_im
#
#         normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
#
#         merge_img = self.transform(merge_img)
#         merge_img = normalize(merge_img)
#         merge_gt = self.transform(np.array(label_alpha, dtype='uint8'))
#         trimap = torch.from_numpy(get_trimap(merge_gt[0].cpu().numpy()))
#
#
#
#         return merge_img, merge_gt, trimap, item
#
#     def get_show_data(self, item: int) -> SHOW_TYPE:
#         merge_img, merge_gt, trimap, item = self.get_val_data(item)
#         return merge_img, merge_gt, item
#
#     def __len__(self):
#         return len(self.merge_files)
