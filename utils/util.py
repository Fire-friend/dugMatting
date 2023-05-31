"""
Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]
Utilization file.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/GFM
Paper link (Arxiv): https://arxiv.org/abs/2010.16188

"""
import importlib
import os
import random
import shutil
from typing import Tuple

import torch
import numpy as np
import cv2
import yaml
from PIL import Image


##########################
### Pure functions
##########################
def extract_pure_name(original_name):
    pure_name, extention = os.path.splitext(original_name)
    return pure_name


def listdir_nohidden(path):
    new_list = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            new_list.append(f)
    new_list.sort()
    return new_list


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def check_if_folder_exists(folder_path):
    return os.path.exists(folder_path)


def refresh_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)


def generate_composite_img(img, alpha_channel):
    b_channel, g_channel, r_channel = cv2.split(img)
    b_channel = b_channel * alpha_channel
    g_channel = g_channel * alpha_channel
    r_channel = r_channel * alpha_channel
    alpha_channel = (alpha_channel * 255).astype(b_channel.dtype)
    img_BGRA = cv2.merge((r_channel, g_channel, b_channel, alpha_channel))
    return img_BGRA


def select_roi(uncertain_map, mode='hist', min_th=None):
    with torch.no_grad():
        roi = []
        _, _, h_, w_ = uncertain_map.shape
        for i in range(len(uncertain_map)):
            if mode == 'hist':
                me = torch.min(uncertain_map[i])
            else:
                me = torch.mean(uncertain_map[i])
            mmax = torch.max(uncertain_map[i])
            # mmin = torch.min(uncertain_map[i])
            roi_index = uncertain_map[i] > me
            hist_map = torch.histc(uncertain_map[i][roi_index].float(), bins=10, min=me, max=mmax)
            threshold_t = me + (mmax - me) * 0.1 * (torch.argmax(hist_map))
            if mode == 'hist':  # mode == 'semantic':
                step = (mmax - me) / len(hist_map)
                max_g = 0
                for t in range(len(hist_map)):
                    # 使用numpy直接对数组进行计算
                    n0 = uncertain_map[i][uncertain_map[i] < t * step + me]
                    n1 = uncertain_map[i][uncertain_map[i] >= t * step + me]
                    w0 = len(n0) / (h_ * w_)
                    w1 = len(n1) / (h_ * w_)
                    u0 = torch.mean(n0) if len(n0) > 0 else 0
                    u1 = torch.mean(n1) if len(n1) > 0 else 0

                    g = w0 * w1 * (u0 - u1) ** 2
                    if g > max_g:
                        max_g = g
                        threshold_t = t * step + me
            roi_index = uncertain_map[i] > threshold_t
            # print(threshold_t)
            if min_th is not None:
                if threshold_t < min_th:
                    roi_index = torch.zeros_like(uncertain_map[i])
                    roi_index = roi_index == 1

            if torch.sum(roi_index) > (h_ * w_ * 0.25):
                roi_index_ = torch.topk(uncertain_map[i].view(1, -1), int(h_ * w_ * 0.25), dim=1).indices
                roi_index = torch.zeros_like(uncertain_map[i].reshape(1, -1))
                roi_index.scatter_(1, roi_index_, 1.)
                roi_index = roi_index.view(1, h_, w_).type(torch.bool)

            roi.append(roi_index.unsqueeze(0))
        roi_index = torch.cat(roi, dim=0)
        return roi_index


import torch.nn.functional as F


def crop_patch(x: torch.Tensor,
               idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               size: int,
               padding: int):
    """
    Crops selected patches from image given indices.

    Inputs:
        x: image (B, C, H, W).
        idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
        size: center size of the patch, also stride of the crop.
        padding: expansion size of the patch.
    Output:
        patch: (P, C, h, w), where h = w = size + 2 * padding.
    """
    if padding != 0:
        x = F.pad(x, (padding,) * 4)
    # Use unfold. Best performance for PyTorch and TorchScript.
    return x.permute(0, 2, 3, 1) \
        .unfold(1, size + 2 * padding, size) \
        .unfold(2, size + 2 * padding, size)[idx[:, 0], idx[:, 1], idx[:, 2]]


def replace_patch(x: torch.Tensor,
                  y: torch.Tensor,
                  idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """
    Replaces patches back into image given index.

    Inputs:
        x: image (B, C, H, W)
        y: patches (P, C, h, w)
        idx: selection indices Tuple[(P,), (P,), (P,)] where the 3 values are (B, H, W) index.

    Output:
        image: (B, C, H, W), where patches at idx locations are replaced with y.
    """
    xB, xC, xH, xW = x.shape
    yB, yC, yH, yW = y.shape
    # Use scatter_nd. Best performance for PyTorch and TorchScript. Replacing patch by patch.
    x = x.view(xB, xC, xH // yH, yH, xW // yW, yW).permute(0, 2, 4, 1, 3, 5)
    x[idx[:, 0], idx[:, 1], idx[:, 2]] = y
    x = x.permute(0, 3, 1, 4, 2, 5).view(xB, xC, xH, xW)
    return x


##########################
### Functions for fusion 
##########################
def collaborative_matting(rosta, glance_sigmoid, focus_sigmoid):
    if rosta == 'TT':
        values, index = torch.max(glance_sigmoid, 1)
        index = index[:, None, :, :].float()
        ### index <===> [0, 1, 2]
        ### bg_mask <===> [1, 0, 0]
        bg_mask = index.clone()
        bg_mask[bg_mask == 2] = 1
        bg_mask = 1 - bg_mask
        ### trimap_mask <===> [0, 1, 0]
        trimap_mask = index.clone()
        trimap_mask[trimap_mask == 2] = 0
        ### fg_mask <===> [0, 0, 1]
        fg_mask = index.clone()
        fg_mask[fg_mask == 1] = 0
        fg_mask[fg_mask == 2] = 1
        focus_sigmoid = focus_sigmoid.cpu()
        trimap_mask = trimap_mask.cpu()
        fg_mask = fg_mask.cpu()
        fusion_sigmoid = focus_sigmoid * trimap_mask + fg_mask
    elif rosta == 'BT':
        values, index = torch.max(glance_sigmoid, 1)
        index = index[:, None, :, :].float()
        fusion_sigmoid = index - focus_sigmoid
        fusion_sigmoid[fusion_sigmoid < 0] = 0
    else:
        values, index = torch.max(glance_sigmoid, 1)
        index = index[:, None, :, :].float()
        fusion_sigmoid = index + focus_sigmoid
        fusion_sigmoid[fusion_sigmoid > 1] = 1
    fusion_sigmoid = fusion_sigmoid.cuda()
    return fusion_sigmoid


def get_masked_local_from_global_test(global_result, local_result):
    weighted_global = np.ones(global_result.shape)
    weighted_global[global_result == 255] = 0
    weighted_global[global_result == 0] = 0
    fusion_result = global_result * (1. - weighted_global) / 255 + local_result * weighted_global
    return fusion_result


def gen_trimap_from_segmap_e2e(segmap):
    trimap = np.argmax(segmap, axis=1)[0]
    trimap = trimap.astype(np.int64)
    trimap[trimap == 1] = 128
    trimap[trimap == 2] = 255
    return trimap.astype(np.uint8)


def gen_bw_from_segmap_e2e(segmap):
    bw = np.argmax(segmap, axis=1)[0]
    bw = bw.astype(np.int64)
    bw[bw == 1] = 255
    return bw.astype(np.uint8)


def save_test_result(save_dir, predict):
    predict = (predict * 255).astype(np.uint8)
    cv2.imwrite(save_dir, predict)


def trim_img(img):
    if img.ndim > 2:
        img = img[:, :, 0]
    return img


def resize_img(ori, img):
    img = cv2.resize(img, ori.shape) * 255.0
    return img


def process_fgbg(ori, mask, is_fg, fgbg_path=None):
    if fgbg_path is not None:
        img = np.array(Image.open(fgbg_path))
    else:
        mask_3 = (mask / 255.0)[:, :, np.newaxis].astype(np.float32)
        img = ori * mask_3 if is_fg else ori * (1 - mask_3)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)
    erode = cv2.erode(fg, kernel, iterations=1)
    trimap = erode * 255 + (dilate - erode) * 128
    return trimap.astype(np.uint8)


def gen_dilate(alpha, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1) * 255
    return dilate.astype(np.uint8)


def gen_erosion(alpha, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    erode = cv2.erode(fg, kernel, iterations=1) * 255
    return erode.astype(np.uint8)


def get_trimap(alpha):
    alpha255 = np.array(alpha * 255, dtype='uint8')
    th, binary = cv2.threshold(alpha255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # alpha255[alpha255 < 0.1] = 0
    # alpha255[alpha255 >= 0.1] = 255

    k_size = int(alpha.shape[0] / 20)
    # k_size = 40
    iterations = 1
    # iterations = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(binary.copy(), kernel, iterations)
    eroded = cv2.erode(binary.copy(), kernel, iterations)
    # cv2.imshow('dilated', dilated)
    # cv2.imshow('eroded', eroded)
    # cv2.imshow('dis', dilated - eroded)
    # cv2.waitKey(500)
    trimap = np.zeros(alpha.shape)
    trimap[eroded == 255] = 1
    trimap[dilated - eroded == 255] = 0.5
    # if np.sum(trimap) == 0:
    #     trimap[alpha != 0] = 0.5
    # plt.imshow(trimap * 255, cmap='gray')
    # plt.show()
    # plt.savefig('./temp.png', cmap='gray')
    return trimap


def get_trimap2(alpha):
    alpha_fg = alpha.copy()
    alpha_bg = alpha.copy()
    alpha_fg[alpha_fg < 1] = 0
    alpha_fg[alpha_fg == 1] = 255
    alpha_bg[alpha_bg > 0] = 255
    alpha_bg = 255 - alpha_bg
    # k_size = random.choice(range(3, 6))
    # iterations = np.random.randint(1, 20)
    # k_size = int(alpha.shape[0] / 50)
    k_size = 15
    iterations = 1
    # iterations = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated_fg = cv2.erode(alpha_fg, kernel, iterations)
    dilated_bg = cv2.erode(alpha_bg, kernel, iterations)
    # plt.imshow(dilated, cmap='gray')
    # plt.show()
    # plt.imshow(eroded, cmap='gray')
    # plt.show()
    # cv2.imshow('dilated', dilated)
    # cv2.imshow('eroded', eroded)
    # cv2.imshow('dis', dilated - eroded)
    # cv2.waitKey(500)
    trimap = np.zeros(alpha.shape)
    trimap.fill(0.5)
    trimap[dilated_fg == 255] = 1
    trimap[dilated_bg == 255] = 0
    trimap[(alpha > 0) * (alpha < 1)] = 0.5
    # if np.sum(trimap) == 0:
    #     trimap[alpha != 0] = 0.5
    # plt.imshow(trimap * 255, cmap='gray')
    # plt.show()
    # plt.savefig('./temp.png', cmap='gray')
    return trimap


def scale_tensor(img, mode='short', size=1024, skip_small=True, inter='bilinear'):
    im_size = img.shape
    if skip_small and im_size[2] < size and im_size[3] < size:
        ratio = 1
    else:
        if mode == 'short':
            ratio = min(im_size[2], im_size[3]) / size
        elif mode == 'long':
            ratio = max(im_size[2], im_size[3]) / size
    n_h = im_size[2] / ratio
    n_w = im_size[3] / ratio
    n_h, n_w = max(int(n_h / 32), 1) * 32, max(int(n_w / 32), 1) * 32

    # img = cv2.resize(img, (n_w, n_h), interpolation=cv2.INTER_LINEAR)
    img = F.interpolate(img, (n_h, n_w), mode=inter)
    # cv2.imwrite('temp.png', img)
    return img


def scale_img(img, mode='short', size=1024, skip_small=True):
    im_size = img.shape
    if skip_small and im_size[0] < size and im_size[1] < size:
        ratio = 1
    else:
        if mode == 'short':
            ratio = min(im_size[0], im_size[1]) / size
        elif mode == 'long':
            ratio = max(im_size[0], im_size[1]) / size
    n_h = im_size[0] / ratio
    n_w = im_size[1] / ratio
    n_h, n_w = max(int(n_h / 32), 1) * 32, max(int(n_w / 32), 1) * 32

    img = cv2.resize(img, (n_w, n_h), interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite('temp.png', img)
    return img


def padding_to_square(im):
    im_size = im.shape
    top = max(int((im_size[1] - im_size[0]) / 2), 0)
    bottom = max(im_size[1] - top - im_size[0], 0)
    left = max(int((im_size[0] - im_size[1]) / 2), 0)
    right = max(im_size[0] - left - im_size[1], 0)
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


def generateRandomPrior(alpha, num_fg=[1, 5], num_bg=[1, 5], size=5):
    size = size / 2 * 2 + 1
    pad = int((size - 1) / 2)
    prior_fg = np.zeros(shape=alpha.shape)
    prior_bg = np.zeros(shape=alpha.shape)

    # nums_fg = random.randint(num_fg[0], num_fg[1])
    nums_fg = np.random.geometric(1 / 6)
    fg_index = np.where(alpha == 1)  # [pad + 1:-pad - 1, pad + 1:-pad - 1]
    if len(fg_index[0]) > 0:
        for i in range(nums_fg):
            idx = random.randint(0, len(fg_index[0]) - 1)
            prior_fg[fg_index[0][idx] + pad + 1 - pad:fg_index[0][idx] + pad + 1 + pad,
            fg_index[1][idx] + pad + 1 - pad:fg_index[1][idx] + pad + 1 + pad] = 1

    prior_fg[alpha != 1] = 0

    # nums_bg = random.randint(num_bg[0], num_bg[1])
    nums_bg = np.random.geometric(1 / 6)
    bg_index = np.where(alpha == 0)  # [pad + 1:-pad - 1, pad + 1:-pad - 1]
    if len(bg_index[0]) > 0:
        for i in range(nums_bg):
            idx = random.randint(0, len(bg_index[0]) - 1)
            prior_bg[bg_index[0][idx] + pad + 1 - pad: bg_index[0][idx] + pad + 1 + pad,
            bg_index[1][idx] + pad + 1 - pad:bg_index[1][idx] + pad + 1 + pad] = -1

    prior_bg[alpha != 0] = 0

    # cv2.imshow('p',np.array(prior * 255, dtype='uint8'))
    # cv2.waitKey(0)
    return prior_fg + prior_bg


def cropRoiRegion(img_with_alpha):
    # img = img_with_alpha[..., :3]
    alpha = img_with_alpha[..., -1].copy()
    alpha[alpha < 0.9] = 0
    alpha[alpha >= 0.9] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # alpha = cv2.erode(alpha, kernel)
    alpha = cv2.dilate(alpha, kernel)

    cnts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    leftX_top = 10000
    leftY_top = 10000
    rightX_bottom = -1
    rightY_bottom = -1
    for cnt in cnts:
        if len(cnt) < 400:
            continue
        rect = cv2.boundingRect(cnt)
        rects.append(rect)
        x, y, w, h = rect
        rightX = x + w
        rightY = y + h
        if x < leftX_top:
            leftX_top = x
        if y < leftY_top:
            leftY_top = y
        if rightX > rightX_bottom:
            rightX_bottom = rightX
        if rightY > rightY_bottom:
            rightY_bottom = rightY

        # if w > W:
        #     W = w
        # if h > H:
        #     H = h
        # im = img[y:y + h, x:x + w]
        # plt.subplot(1,2,1)
        # plt.imshow(img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(im)
        # plt.savefig('sszz.png')
        # print()
    H = max(0, rightY_bottom - leftY_top)
    W = max(0, rightX_bottom - leftX_top)
    roi = img_with_alpha[leftY_top:leftY_top + H, leftX_top:leftX_top + W]

    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(roi[..., :3])
    # plt.savefig('sszz.png')

    if roi.shape[0] * roi.shape[1] < 2500:
        return img_with_alpha
    else:
        return roi
    # print(im.shape)
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(im)
    # plt.savefig('sszz.png')
    # print()


import matplotlib.pyplot as plt


def show_tensor(show_tensor, mode='gray'):
    plt.axis('off')
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    if mode is not None:
        plt.imshow(show_tensor.detach().cpu().numpy(), cmap=mode)
    else:
        plt.imshow(show_tensor.detach().cpu().numpy())
    plt.show()


def save_tensor(save_tensor, mode='gray'):
    # plt.imsave(save_tensor.detach().cpu().numpy())

    plt.axis('off')
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # 保存图片，cmap为调整配色方案
    if mode is not None:
        plt.imshow(save_tensor.detach().cpu().numpy(), cmap=mode)
    else:
        plt.imshow(save_tensor.detach().cpu().numpy())
    plt.savefig("./image.png")


# 获取动态导入模块中,公共的do_something
def getPackByNameUtil(py_name, object_name):
    module_object = importlib.import_module(py_name)
    object = getattr(module_object, object_name)
    return object


def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    data = yaml.load(file_data, yaml.FullLoader)
    file.close()
    return data


def set_yaml_to_args(args, dict: dict):
    for key, val in dict.items():
        args.__setattr__(key, val)

    head_save = './checkSave/' + str(args.model) + '/' + str(args.data_set) + '/' + str(args.save_file)

    # args.log_path = head_save + '/' + 'log.txt'
    args.log_path = head_save + '/log/'
    args.save_path_img = head_save + '/temp_results/'
    args.save_path_model = head_save + '/checkpoint/'

    if not os.path.exists(args.save_path_img):
        os.makedirs(args.save_path_img)
    if not os.path.exists(args.save_path_model):
        os.makedirs(args.save_path_model)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)


def freeze_layer(layer):
    for name, value in layer.named_parameters():
        value.requires_grad = False


def norm_batch(tensor):
    # assert torch.isnan(tensor).any() == False, 'error Nan tensor'
    n, c, h, w = tensor.shape
    min_v = torch.min(tensor.view(n, c, -1), dim=2, keepdim=True)[0].unsqueeze(2)
    max_v = torch.max(tensor.view(n, c, -1), dim=2, keepdim=True)[0].unsqueeze(2)
    return (tensor - min_v) / (max_v - min_v + 1e-10)


def get_masked_local_from_global(global_sigmoid, local_sigmoid):
    values, index = torch.max(global_sigmoid, 1)
    index = index[:, None, :, :].float()
    ### index <===> [0, 1, 2]
    ### bg_mask <===> [1, 0, 0]
    bg_mask = index.clone()
    bg_mask[bg_mask == 2] = 1
    bg_mask = 1 - bg_mask
    ### trimap_mask <===> [0, 1, 0]
    trimap_mask = index.clone()
    trimap_mask[trimap_mask == 2] = 0
    ### fg_mask <===> [0, 0, 1]
    fg_mask = index.clone()
    fg_mask[fg_mask == 1] = 0
    fg_mask[fg_mask == 2] = 1
    fusion_sigmoid = local_sigmoid * trimap_mask + fg_mask
    return fusion_sigmoid


def getOneHot(prior, cls=4):
    prior[prior == -1] = 3
    prior[prior == 0.5] = 2
    prior = F.one_hot(prior[:, 0].long(), num_classes=cls)
    prior = prior.permute([0, 3, 1, 2])
    prior = prior[:, 1:]
    return prior


def smooth_xy(x_value: np.ndarray, y_value: np.ndarray):
    from scipy.interpolate import interp1d
    cubic_interploation_model = interp1d(x_value, y_value, kind="cubic")
    x_smooth = np.linspace(x_value.min(), x_value.max(), 500)
    y_smooth = cubic_interploation_model(x_smooth)
    return x_smooth, y_smooth

def moe_nig(u1, la1, alpha1, beta1, u2, la2, alpha2, beta2):
    # Eq. 9
    # u = (la1 * u1 + u2 * la2) / (la1 + la2)

    u = u1
    index = (la1 - la2 < -80) * (la1 < 5)
    print(torch.sum(index))
    u[index] = u2[index]
    la = la1 + la2
    alpha = alpha1 + alpha2 + 0.5
    beta = beta1 + beta2 + 0.5 * (la1 * (u1 - u) ** 2 + la2 * (u2 - u) ** 2)
    return u, la, alpha, beta