import numpy as np
import scipy
import torch


def matte_sad(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_sad = np.sum(np.abs(pred_matte - gt_matte))
    return error_sad


def matte_sad_torch(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_sad = torch.sum(torch.abs(pred_matte - gt_matte))
    return error_sad


def matte_mad_torch(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_mad = torch.mean(torch.abs(pred_matte - gt_matte))
    return error_mad


def matte_mad(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_mad = np.mean(np.abs(pred_matte - gt_matte))
    return error_mad


def matte_mse_torch(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_mse = torch.mean(torch.pow(pred_matte - gt_matte, 2))
    return error_mse


def matte_mse(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_mse = np.mean(np.power(pred_matte - gt_matte, 2))
    return error_mse


def matte_grad(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    # alpha matte 的归一化梯度，标准差 =1.4，1 阶高斯导数的卷积
    predict_grad = scipy.ndimage.filters.gaussian_filter(pred_matte, 1.4, order=1)
    gt_grad = scipy.ndimage.filters.gaussian_filter(gt_matte, 1.4, order=1)
    error_grad = np.sum(np.power(predict_grad - gt_grad, 2))
    return error_grad


def matte_grad_torch(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    # alpha matte 的归一化梯度，标准差 =1.4，1 阶高斯导数的卷积
    predict_grad = scipy.ndimage.filters.gaussian_filter(pred_matte, 1.4, order=1)
    gt_grad = scipy.ndimage.filters.gaussian_filter(gt_matte, 1.4, order=1)
    error_grad = torch.sum(torch.pow(predict_grad - gt_grad, 2))
    return error_grad


def calculate_sad_fgbg(predict, alpha, trimap):
    sad_diff = np.abs(predict - alpha)
    weight_fg = np.zeros(predict.shape)
    weight_bg = np.zeros(predict.shape)
    weight_trimap = np.zeros(predict.shape)
    trimap[trimap == 1] = 255
    trimap[trimap == 0.5] = 128
    trimap[trimap == 0] = 0
    weight_fg[trimap == 255] = 1.
    weight_bg[trimap == 0] = 1.
    weight_trimap[trimap == 128] = 1.
    sad_fg = np.sum(sad_diff * weight_fg) / 1000
    sad_bg = np.sum(sad_diff * weight_bg) / 1000

    sad_trimap = np.sum(sad_diff * weight_trimap) / 1000
    return sad_fg, sad_bg, sad_trimap


def calculate_sad_fgbg_torch(predict, alpha, trimap):
    sad_diff = torch.abs(predict - alpha)
    weight_fg = torch.zeros_like(predict)
    weight_bg = torch.zeros_like(predict)
    weight_trimap = torch.zeros_like(predict)
    # trimap = trimap.numpy()
    # trimap[trimap == 1] = 255
    # trimap[trimap == 0.5] = 128
    # trimap[trimap == 0] = 0
    weight_fg[trimap == 1] = 1.
    weight_bg[trimap == 0] = 1.
    weight_trimap[trimap == 0.5] = 1.
    sad_fg = torch.sum(sad_diff * weight_fg)
    sad_bg = torch.sum(sad_diff * weight_bg)
    sad_trimap = torch.sum(sad_diff * weight_trimap)
    return sad_fg, sad_bg, sad_trimap


def compute_gradient_whole_image(pd, gt):
    from scipy.ndimage import gaussian_filter

    pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
    pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
    gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
    gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
    pd_mag = np.sqrt(pd_x ** 2 + pd_y ** 2)
    gt_mag = np.sqrt(gt_x ** 2 + gt_y ** 2)

    error_map = np.square(pd_mag - gt_mag)
    loss = np.sum(error_map) / 10
    return loss


import numpy as np
import cv2
from scipy import ndimage


def compute_connectivity_error(pred, target, step=0.1, trimap=None):
    """
    计算预测alpha和真实alpha的连通度
    :param pred: 预测alpha，单通道图像， 0-255，(h, w)
    :param target: 真实alpha，单通道图像 0-255，(h, w)
    :param step: 默认为0.1
    :param trimap(optional): 如果提供，只计算trimap128区域的连通度；单通道图像，只有三个值，0-128-255，(h,w)
    :return: loss
    """
    pred = np.array(pred, dtype=float) / 255
    target = np.array(target, dtype=float) / 255

    h, w = pred.shape

    thresh_steps = np.linspace(0, 1, int(1 / step + 1))
    l_map = np.ones(shape=pred.shape) * -1
    dist_maps = np.zeros((h, w, len(thresh_steps)))

    for ii in range(2, len(thresh_steps)):
        pred_alpha_thresh = pred >= thresh_steps[ii]
        target_alpha_thresh = target >= thresh_steps[ii]

        binary = np.array(pred_alpha_thresh & target_alpha_thresh, dtype='uint8')
        num, labels = cv2.connectedComponents(binary, connectivity=4)
        la, la_num = np.unique(labels[labels != 0], return_counts=True)
        if len(la_num) > 0:
            max_la = la[np.argmax(la_num)]
        else:
            max_la = -1
        omega = np.zeros((h, w))
        omega[labels == max_la] = 1
        flag = (l_map == -1) & (omega == 0)
        l_map[flag == 1] = thresh_steps[ii - 1]
        dist_maps[:, :, ii] = ndimage.distance_transform_edt(omega)
        dist_maps[:, :, ii] = dist_maps[:, :, ii] / (np.max(dist_maps[:, :, ii]) + 1e-10)

    l_map[l_map == -1] = 1
    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d > 0.15)
    target_phi = 1 - target_d * (target_d > 0.15)
    if trimap is None:
        loss = np.sum(np.abs(pred_phi - target_phi))
    else:
        loss = np.sum(np.abs(pred_phi - target_phi) * (trimap == 128))
    return loss / 1000


def computeAllMatrix(matte, label_alpha, trimap):
    error_sad = matte_sad_torch(matte, label_alpha)
    error_mad = matte_mad_torch(matte, label_alpha)
    error_mse = matte_mse_torch(matte, label_alpha)
    error_grad = compute_gradient_whole_image(matte.squeeze(0).squeeze(0).detach().cpu().numpy(),
                                              label_alpha.squeeze(0).squeeze(0).detach().cpu().numpy())
    sad_fg, sad_bg, sad_tran = calculate_sad_fgbg_torch(matte, label_alpha, trimap)
    # conn = compute_connectivity_error(matte.squeeze(0).squeeze(0).detach().cpu().numpy()*255,
    #                                   label_alpha.squeeze(0).squeeze(0).detach().cpu().numpy()*255)
    conn = 0
    return error_sad, error_mad, error_mse, torch.tensor(error_grad).cuda(), sad_fg, sad_bg, sad_tran, conn
