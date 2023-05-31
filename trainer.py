import math

import scipy
from kornia.losses import ssim
from scipy.ndimage import grey_dilation, grey_erosion
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from evaluate import *
from utils.loss_util import compute_bfd_loss_prior, muti_bce_loss_fusion, compute_bfd_loss, compute_bfd_loss_seg, \
    compute_bfd_loss_mat, compute_bfd_loss_mat_single, compute_bfd_loss_mat_single2, loss_function_SHM, loss_FBDM_img


class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)),
            nn.Conv2d(channels, channels, self.kernel_size,
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor
        Returns:
            torch.Tensor: Blurred version of the input
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()

        return self.op(x)

    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))


def supervised_training_iter(
        net, image, trimap=None, gt_matte=None, prior=None, prior_trimap=None,
        mode='modnet', blurer=None, fg=None, bg=None, args=None, epoch=None):
    if prior is not None and prior_trimap is not None:
        prior = prior.cuda().float()
        prior_trimap = prior_trimap.cuda().float()
    # forward the model
    if mode == 'modnet':
        semantic_scale = 10.0
        detail_scale = 10.0
        matte_scale = 1.0

        # calculate the boundary gt_matte from the trimap
        boundaries = (trimap < 0.5) + (trimap > 0.5)
        with autocast():
            pred_semantic, pred_detail, pred_matte = net(image, False)

            # calculate the semantic loss
            gt_semantic = F.interpolate(gt_matte, scale_factor=1 / 16, mode='bilinear')
            gt_semantic = blurer(gt_semantic)
            semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
            semantic_loss = semantic_scale * semantic_loss

            # calculate the detail loss
            if pred_detail.dtype == torch.float16:
                trimap = trimap.half()
                gt_matte = gt_matte.half()
            pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
            gt_detail = torch.where(boundaries, trimap, gt_matte)
            detail_loss = F.l1_loss(pred_boundary_detail, gt_detail)
            detail_loss = detail_scale * detail_loss

            # calculate the matte loss
            pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
            matte_l1_loss = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_detail)
            matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
                                       + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_detail)
            matte_loss = matte_l1_loss + matte_compositional_loss
            matte_loss = matte_scale * matte_loss

            # calculate the final loss, backward the loss, and update the model
            loss = semantic_loss + detail_loss + matte_loss

            return loss, semantic_loss, detail_loss, matte_loss
    elif mode == 'gfm':
        with autocast():
            # pred_semantic, pred_detail, pred_matte, pred_source0, pred_source1, pred_source2, pred_source3 = net(image)
            pred_semantic, pred_detail, pred_matte = net(image)

            loss_global = get_crossentropy_loss(3, trimap, pred_semantic[:trimap.shape[0]])
            loss_local = get_alpha_loss(pred_detail[:trimap.shape[0]], gt_matte, trimap)  # + get_laplacian_loss(
            # pred_detail[:trimap.shape[0]], gt_matte, trimap)

            loss_fusion_alpha = get_alpha_loss_whole_img(pred_matte[:trimap.shape[0]],
                                                         gt_matte)  # + get_laplacian_loss_whole_img(pred_matte[:trimap.shape[0]], gt_matte)

            loss_fusion_comp = get_composition_loss_whole_img(image[:trimap.shape[0]], gt_matte,
                                                              pred_matte[:trimap.shape[0]])

            loss = 0.25 * loss_global + 0.25 * loss_local + 0.25 * loss_fusion_alpha + 0.25 * loss_fusion_comp  # + 0.1 * loss_reconstr
            # loss = loss_local
        return loss, loss_global, loss_local, loss_fusion_alpha, loss_fusion_comp
    elif mode == 'bfd':
        with autocast():
            fg_out_list, bg_out_list, detail_out_fg_list, detail_out_bg_list, prior_fg, prior_bg, out = net(image,
                                                                                                            prior=prior)
            if args.in_w_prior:
                loss, fg_out_loss, bg_out_loss, detail_out_loss, prior_loss_sum, matte_loss = compute_bfd_loss_prior(
                    fg_out_list, bg_out_list, out,
                    gt_matte, image, trimap, fg, bg)
            else:
                loss, fg_out_loss, bg_out_loss, detail_out_loss, prior_loss_sum, matte_loss = compute_bfd_loss_mat(
                    fg_out_list=fg_out_list, bg_out_list=bg_out_list, detail_out_list_fg=detail_out_fg_list,
                    detail_out_list_bg=detail_out_bg_list, out=out,
                    gt_matte=gt_matte, image=image, trimap=trimap, fg=fg, bg=bg, prior_fg=prior_fg, prior_bg=prior_bg,
                    epoch=epoch)

        return loss, fg_out_loss, bg_out_loss, detail_out_loss, prior_loss_sum, matte_loss

        # with autocast():
        #     out_f, out_b, out_d, prior_fg, prior_bg, out = net(image, prior=prior)
        #     if args.in_w_prior:
        #         loss, fg_loss_sum, bg_loss_sum, matte_loss_sum = compute_bfd_loss_prior(
        #             out_f, out_b, out_d, out,
        #             gt_matte, image, trimap, fg, bg)
        #     else:
        #         loss, fg_loss_sum, bg_loss_sum, detail_loss_sum, matte_loss_sum, prior_loss_sum = compute_bfd_loss_mat_single2(
        #             out_f=out_f, out_b=out_b, out_d=out_d, out=out,
        #             gt_matte=gt_matte, image=image, trimap=trimap, fg=fg, bg=bg, prior_fg=prior_fg, prior_bg=prior_bg)
        # return loss, fg_loss_sum, bg_loss_sum, detail_loss_sum, matte_loss_sum, prior_loss_sum
    elif mode == 'u2net' or mode == 'u2netp':
        if prior is not None:
            image = torch.cat([image, prior], dim=1)
        with autocast():
            d0, d1, d2, d3, d4, d5, d6 = net(image)
            loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6,
                                                                                         gt_matte, trimap)
        return loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss
    elif mode == 'SHM':
        with autocast():
            trimap_pre, alpha_pre = net(image)
            loss, L_alpha, L_composition, L_cross = loss_function_SHM(image,
                                                                      trimap_pre,
                                                                      trimap,
                                                                      alpha_pre,
                                                                      gt_matte)
        return loss, L_alpha, L_composition, L_cross
    elif mode == 'FBDM_img':
        with autocast():
            I_sigmoid, fusion_sigmoid = net(image)
            loss, f_loss, b_loss, t_loss, m_loss = loss_FBDM_img(I_sigmoid, fusion_sigmoid, gt_matte, image)
        return loss, f_loss, b_loss, t_loss, m_loss
    else:
        assert 'the mode is not supprot'


def soc_adaptation_iter(
        net, backup_modnet, optimizer, image,
        soc_semantic_scale=10.0, soc_detail_scale=10.0):
    """ Self-Supervised sub-objective consistency (SOC) adaptation iteration of net
    This function fine-tunes net for one iteration in an unlabeled dataset.
    Note that SOC can only fine-tune a converged net, i.e., net that has been
    trained in a labeled dataset.

    Arguments:
        net (torch.nn.Module): instance of net
        backup_modnet (torch.nn.Module): backup of the trained net
        optimizer (torch.optim.Optimizer): optimizer for self-supervised SOC
        image (torch.autograd.Variable): input RGB image
                                         its pixel values should be normalized
        soc_semantic_scale (float): scale of the SOC semantic loss
                                    NOTE: please adjust according to your dataset
        soc_detail_scale (float): scale of the SOC detail loss
                                  NOTE: please adjust according to your dataset

    Returns:
        soc_semantic_loss (torch.Tensor): loss of the semantic SOC
        soc_detail_loss (torch.Tensor): loss of the detail SOC

    Example:
        import copy
        import torch
        from src.models.net import net
        from src.trainer import soc_adaptation_iter

        bs = 1          # batch size
        lr = 0.00001    # learn rate
        epochs = 10     # total epochs

        net = torch.nn.DataParallel(net()).cuda()
        net = LOAD_TRAINED_CKPT()    # NOTE: please finish this function

        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99))
        dataloader = CREATE_YOUR_DATALOADER(bs)     # NOTE: please finish this function

        for epoch in range(0, epochs):
            backup_modnet = copy.deepcopy(net)
            for idx, (image) in enumerate(dataloader):
                soc_semantic_loss, soc_detail_loss = \
                    soc_adaptation_iter(net, backup_modnet, optimizer, image)
    """

    global blurer

    # set the backup model to eval mode
    backup_modnet.eval()

    # set the main model to train mode and freeze its norm layers
    net.train()
    net.module.freeze_norm()

    # clear the optimizer
    optimizer.zero_grad()

    # forward the main model
    pred_semantic, pred_detail, pred_matte = net(image, False)

    # forward the backup model
    with torch.no_grad():
        _, pred_backup_detail, pred_backup_matte = backup_modnet(image, False)

    # calculate the boundary gt_matte from `pred_matte` and `pred_semantic`
    pred_matte_fg = (pred_matte.detach() > 0.1).float()
    pred_semantic_fg = (pred_semantic.detach() > 0.1).float()
    pred_semantic_fg = F.interpolate(pred_semantic_fg, scale_factor=16, mode='bilinear')
    pred_fg = pred_matte_fg * pred_semantic_fg

    n, c, h, w = pred_matte.shape
    np_pred_fg = pred_fg.data.cpu().numpy()
    np_boundaries = np.zeros([n, c, h, w])
    for sdx in range(0, n):
        sample_np_boundaries = np_boundaries[sdx, 0, ...]
        sample_np_pred_fg = np_pred_fg[sdx, 0, ...]

        side = int((h + w) / 2 * 0.05)
        dilated = grey_dilation(sample_np_pred_fg, size=(side, side))
        eroded = grey_erosion(sample_np_pred_fg, size=(side, side))

        sample_np_boundaries[np.where(dilated - eroded != 0)] = 1
        np_boundaries[sdx, 0, ...] = sample_np_boundaries

    boundaries = torch.tensor(np_boundaries).float().cuda()

    # sub-objectives consistency between `pred_semantic` and `pred_matte`
    # generate pseudo ground truth for `pred_semantic`
    downsampled_pred_matte = blurer(F.interpolate(pred_matte, scale_factor=1 / 16, mode='bilinear'))
    # downsampled_pred_matte = pred_matte
    pseudo_gt_semantic = downsampled_pred_matte.detach()
    pseudo_gt_semantic = pseudo_gt_semantic * (pseudo_gt_semantic > 0.01).float()

    # generate pseudo ground truth for `pred_matte`
    pseudo_gt_matte = pred_semantic.detach()
    pseudo_gt_matte = pseudo_gt_matte * (pseudo_gt_matte > 0.01).float()

    # calculate the SOC semantic loss
    soc_semantic_loss = F.binary_cross_entropy(pred_semantic, pseudo_gt_semantic) + F.binary_cross_entropy(
        downsampled_pred_matte,
        pseudo_gt_matte)
    soc_semantic_loss = soc_semantic_scale * torch.mean(soc_semantic_loss)

    # NOTE: using the formulas in our paper to calculate the following losses has similar results
    # sub-objectives consistency between `pred_detail` and `pred_backup_detail` (on boundaries only)
    backup_detail_loss = boundaries * F.l1_loss(pred_detail, pred_backup_detail, reduction='none')
    backup_detail_loss = torch.sum(backup_detail_loss, dim=(1, 2, 3)) / (torch.sum(boundaries, dim=(1, 2, 3)) + 1e-10)
    backup_detail_loss = torch.mean(backup_detail_loss)

    # sub-objectives consistency between pred_matte` and `pred_backup_matte` (on boundaries only)
    backup_matte_loss = boundaries * F.l1_loss(pred_matte, pred_backup_matte, reduction='none')
    backup_matte_loss = torch.sum(backup_matte_loss, dim=(1, 2, 3)) / (torch.sum(boundaries, dim=(1, 2, 3)) + 1e-10)
    backup_matte_loss = torch.mean(backup_matte_loss)

    soc_detail_loss = soc_detail_scale * (backup_detail_loss + backup_matte_loss)

    # calculate the final loss, backward the loss, and update the model
    loss = soc_semantic_loss + soc_detail_loss

    loss.backward()
    optimizer.step()

    return soc_semantic_loss, soc_detail_loss
