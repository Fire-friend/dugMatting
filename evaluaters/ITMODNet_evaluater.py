# import time

import cv2
import kornia.augmentation
import numpy as np
import torch

# from utils.MGmattingUtil import moe_nig
from utils.eval import computeAllMatrix
import torch.nn.functional as F
# from skimage.transform import resize
# from utils.uncertainty_eva import CalibratedRegression
# import matplotlib.pyplot as plt

from utils.util import show_tensor, select_roi, smooth_xy, moe_nig

precision_global = 0
recall_global = 0
index_global = 0
y_pred = None
y_label = None
Auc = 0


def ITMODNet_Evaluater(net, input, gt, trimap, fusion=False, interac=2, instance_map=None):
    global precision_global, recall_global, index_global, y_pred, y_label, Auc
    instance_map = torch.zeros_like(input)[:, 0:1]
    # user_map = torch.zeros_like(input)[:, 0:1]
    last_output = None
    for it in range(interac):
        input_temp = torch.cat([input, instance_map], dim=1)
        last_out = net(input_temp, inference=True)
        cur_matte = last_out[-1]

        if fusion and last_output is not None:
            last_output = \
                moe_nig(last_out[-1], last_out[2], last_out[3], last_out[4],
                        last_output[0], last_output[1], last_output[2], last_output[3])
            fusion_matte = last_output[0]
        else:
            fusion_matte = cur_matte
            # last_output = last_out[-1], last_out[2], last_out[3], last_out[4]

        if fusion_matte.shape[2] != gt.shape[2]:
            fusion_matte = F.interpolate(fusion_matte, (gt.shape[2], gt.shape[3]), mode='bilinear')
        if it < interac:
            # uncertainty
            with torch.no_grad():
                un = last_out[4] / (last_out[2] * (last_out[3] - 1))
                alea_un = last_out[4] / ((last_out[3] - 1))
                alea_var = (last_out[4] ** 2) / (((last_out[3] - 1) ** 2) * (last_out[3] - 2))
                roi = select_roi(alea_var)
                alea_un[roi] = 0
                un = un.squeeze(1)
                alea_un = alea_un.squeeze(1)
                last_un = un
                # un[trimap[:, 0, :, :] == 0.5] = 0
                n, h, w = un.shape
                temp_gt = F.interpolate(gt, (h, w))
                temp_gt[(temp_gt > 0) * (temp_gt < 1)] = 0.5
                patch_size = 10
                pad_h = h % patch_size
                pad_w = w % patch_size
                patch_num_h = h // patch_size
                patch_num_w = w // patch_size
                N = 10
                un = un[:, pad_h // 2:pad_h // 2 + h - pad_h, pad_w // 2:pad_w // 2 + w - pad_w].reshape(n,
                                                                                                         patch_num_h,
                                                                                                         h // patch_num_h,
                                                                                                         patch_num_w,
                                                                                                         w // patch_num_w)
                un = un.permute([0, 1, 3, 2, 4]).reshape(n, patch_num_h * patch_num_w, -1)
                patch_un = torch.mean(un, dim=2)
                temp_gt = temp_gt[:, :, pad_h // 2:pad_h // 2 + h - pad_h, pad_w // 2:pad_w // 2 + w - pad_w].reshape(n,
                                                                                                                      patch_num_h,
                                                                                                                      h // patch_num_h,
                                                                                                                      patch_num_w,
                                                                                                                      w // patch_num_w)
                temp_gt = temp_gt.permute([0, 1, 3, 2, 4]).reshape(n, patch_num_h * patch_num_w, -1)
                patch_gt = torch.mean(temp_gt, dim=2)

                des_index = torch.argsort(patch_un, dim=1, descending=True)
                user_map = torch.zeros_like(patch_un)  # (n,256)
                for i in range(len(user_map)):
                    # user_map[i, des_index[i, :10]] = 1
                    gt_map = patch_gt[i, des_index[i, :N]]
                    user_map[i, des_index[i, :N][gt_map == 1]] = 1
                    user_map[i, des_index[i, :N][gt_map == 0]] = -1
                    user_map[i, des_index[i, :N][(gt_map > 0) * (gt_map < 1)]] = 0.5

                user_map = user_map.reshape(n, 1, patch_num_h, patch_num_w)
                user_map = F.interpolate(user_map, (h - pad_h, w - pad_w))
                user_map = F.pad(user_map, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                                 "constant", 0)
                new_user_map = F.interpolate(instance_map, (h, w))
                new_user_map[user_map != 0] = user_map[user_map != 0]
                instance_map = new_user_map
        else:
            new_user_map = torch.zeros_like(gt)
            last_un = torch.zeros_like(gt).squeeze(0)
            alea_un = torch.zeros_like(gt).squeeze(0)
            alea_var = torch.zeros_like(gt)

    error_sad, error_mad, error_mse, error_grad, sad_fg, sad_bg, sad_tran, conn = computeAllMatrix(fusion_matte,
                                                                                                   gt,
                                                                                                   trimap)
    input = F.interpolate(input, (gt.shape[2], gt.shape[3]))
    new_user_map = F.interpolate(new_user_map, (gt.shape[2], gt.shape[3]))
    last_un = F.interpolate(last_un.unsqueeze(0), (gt.shape[2], gt.shape[3]), mode='bilinear')[0]
    alea_un = F.interpolate(alea_un.unsqueeze(0), (gt.shape[2], gt.shape[3]), mode='bilinear')[0]
    alea_var = F.interpolate(alea_var, (gt.shape[2], gt.shape[3]), mode='bilinear')[0]
    return error_sad, error_mad, error_mse, error_grad, sad_fg, sad_bg, sad_tran, conn, last_un, alea_un, alea_var, input, new_user_map, fusion_matte, fusion_matte


def ITMODNet_Shower(net, input, gt):
    input = torch.cat([input, torch.zeros_like(gt)], dim=1)
    out = net(input)
    matte = out[-1]
    return matte, input
