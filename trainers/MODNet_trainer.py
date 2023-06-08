from torch.cuda.amp import autocast

from utils.evaluate import *


def MODNet_Trainer(
        net, image, trimap=None, gt_matte=None, sp=None, instance_map=None, user_map=None,
        mode='modnet', blurer=None, fg=None, bg=None, args=None, epoch=None, cur_step=None, total_step=None):
    # forward the model
    semantic_scale = 10.0
    detail_scale = 10.0
    matte_scale = 1.0

    # calculate the boundary gt_matte from the trimap
    boundaries = (trimap < 0.5) + (trimap > 0.5)
    with autocast():
        pred_semantic, pred_detail, pred_matte = net(image, False)

        # calculate the semantic loss
        gt_semantic = F.interpolate(gt_matte, scale_factor=1 / 16, mode='bilinear')
        # gt_semantic = blurer(gt_semantic)
        gt_semantic = kornia.filters.gaussian_blur2d(gt_semantic, (3, 3), (0.8, 0.8))
        semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
        semantic_loss = semantic_scale * semantic_loss

        # calculate the detail loss
        trimap = trimap.type(pred_detail.dtype)
        gt_matte = gt_matte.type(pred_detail.dtype)
        pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
        gt_detail = torch.where(boundaries, trimap, gt_matte)
        detail_loss = F.l1_loss(pred_boundary_detail, gt_detail)
        detail_loss = detail_scale * detail_loss

        # calculate the matte loss
        pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
        matte_l1_loss = F.mse_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_detail)
        matte_compositional_loss = F.mse_loss(image * pred_matte, image * gt_matte) \
                                   + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_detail)
        matte_loss = matte_l1_loss + matte_compositional_loss
        matte_loss = matte_scale * matte_loss

        # calculate the final loss, backward the loss, and update the model
        loss = semantic_loss + detail_loss + matte_loss

    return {'loss': loss,
            'l_s': semantic_loss,
            'l_d': detail_loss,
            'l_m': matte_loss}
