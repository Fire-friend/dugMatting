import math

import kornia
import scipy
from kornia.losses import ssim
from scipy.ndimage import grey_dilation, grey_erosion
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from evaluate import *
from utils.loss_util import compute_bfd_loss_prior, muti_bce_loss_fusion, compute_bfd_loss, compute_bfd_loss_seg, \
    compute_bfd_loss_mat, compute_bfd_loss_mat_single, compute_bfd_loss_mat_single2, loss_function_SHM, loss_FBDM_img, \
    mse_loss_SL, criterion_nig
from utils.util import show_tensor


def compute_loss(image, gt_matte, trimap, pred_semantic, pred_detail, pred_matte, pred_la, pred_alpha, pred_beta, epoch,
                 args, semantic_scale, detail_scale, matte_scale):
    boundaries = (trimap < 0.5) + (trimap > 0.5)
    # gt_semantic = gt_matte.squeeze(1).clone()
    # gt_semantic[(gt_semantic > 0) * (gt_semantic < 1)] = 255
    # gt_semantic = gt_semantic.long()
    gt_semantic = F.interpolate(gt_matte, scale_factor=1 / 16, mode='bilinear')
    semantic_loss = F.mse_loss(gt_semantic, pred_semantic)
    semantic_loss = semantic_scale * semantic_loss.mean()

    # calculate the detail loss
    trimap = trimap.type(pred_detail.dtype)
    gt_matte = gt_matte.type(pred_detail.dtype)
    pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
    gt_detail = torch.where(boundaries, trimap, gt_matte)
    detail_loss = F.l1_loss(pred_boundary_detail, gt_detail)
    detail_loss = detail_scale * detail_loss

    # calculate the matte loss
    # pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
    # matte_l1_loss = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_detail)
    # matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
    #                            + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_detail)
    # matte_loss = matte_l1_loss + matte_compositional_loss
    # matte_loss = matte_scale * matte_loss

    matte_loss = criterion_nig(pred_matte, pred_la, pred_alpha, pred_beta, gt_matte, step=epoch, totalStep=args.epoch)
    matte_loss = matte_loss.mean()
    return semantic_loss, detail_loss, matte_loss


def ITMODNet_Trainer(
        net,
        image,
        trimap=None,
        gt_matte=None,
        user_map=None,
        mode=None,
        fg=None,
        bg=None,
        args=None,
        epoch=None,
        cur_step=None,
        total_step=None):
    # forward the model
    semantic_scale = 10.0
    detail_scale = 10.0
    matte_scale = 1.0

    # show
    # im = kornia.augmentation.Denormalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])(image)
    # show_tensor(im.permute([0, 2, 3, 1])[0])
    # show_tensor(gt_matte[0][0], mode='gray')
    # show_tensor(trimap[0][0], mode='gray')
    # # show_tensor(instance_map[0][0], mode='gray')
    # temp_user = user_map[0][0]
    # show_user = im.permute([0, 2, 3, 1])[0]
    # show_user[temp_user == -1, :] = torch.tensor([[1, 0, 0]]).float().cuda()
    # show_user[temp_user == 0.5, :] = torch.tensor([[0, 1, 0]]).float().cuda()
    # show_user[temp_user == 1, :] = torch.tensor([[0, 0, 1]]).float().cuda()
    # show_tensor(show_user)
    # show_tensor(show_user * 0.3 + 0.7 * instance_map[0][0].unsqueeze(2))

    with autocast():
        # first forward
        input = torch.cat([image, user_map], dim=1)  # , user_map
        pred_semantic, pred_detail, pred_la, pred_alpha, pred_beta, pred_matte = net(input, False)
        # calculate the final loss, backward the loss, and update the model
        semantic_loss, detail_loss, matte_loss = compute_loss(image, gt_matte, trimap, pred_semantic, pred_detail,
                                                              pred_matte, pred_la, pred_alpha, pred_beta, epoch, args,
                                                              semantic_scale, detail_scale, matte_scale)

        # user_semantic_loss, user_detail_loss, user_matte_loss = compute_loss(image, gt_matte, trimap,
        #                                                                      user_pred_semantic, user_pred_detail,
        #                                                                      user_pred_matte, epoch, args,
        #                                                                      semantic_scale,
        #                                                                      detail_scale, matte_scale)
        # user_loss = 0.5 * user_semantic_loss + user_detail_loss + user_matte_loss
        loss = semantic_loss + detail_loss + matte_loss  # + cluster_loss
        loss = loss  # + user_loss

    return {'loss': loss,
            'l_s': semantic_loss,
            'l_d': detail_loss,
            'l_m': matte_loss,
            # 'u_s': user_semantic_loss,
            # 'l_c': cluster_loss
            }
