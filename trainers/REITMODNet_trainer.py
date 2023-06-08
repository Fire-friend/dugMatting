from torch.cuda.amp import autocast

from utils.evaluate import *
from utils.loss_util import criterion_nig


def disLossSigma(pred, gt, logsigma):
    loss = F.l1_loss(pred, gt, reduction='none')
    loss = loss / (2 * torch.exp(logsigma) ** 2) + logsigma
    loss = loss.mean()
    return loss

def compute_loss(image=None, gt_matte=None, trimap=None, pred_semantic=None, pred_detail=None, pred_matte=None,
                 pred_la=None, pred_alpha=None, pred_beta=None, epoch=None,
                 args=None, semantic_scale=None, detail_scale=None, matte_scale=None, logsigma=None):
    boundaries = (trimap < 0.5) + (trimap > 0.5)
    # gt_semantic = gt_matte.squeeze(1).clone()
    # gt_semantic[(gt_semantic > 0) * (gt_semantic < 1)] = 255
    # gt_semantic = gt_semantic.long()
    # gt_semantic = F.interpolate(gt_matte, scale_factor=1 / 16, mode='bilinear')
    # semantic_loss = F.mse_loss(gt_semantic, pred_semantic)
    # semantic_loss = semantic_scale * semantic_loss.mean()

    # calculate the detail loss
    trimap = trimap.type(pred_matte.dtype)
    gt_matte = gt_matte.type(pred_matte.dtype)
    # pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
    gt_detail = torch.where(boundaries, trimap, gt_matte)
    # detail_loss = F.l1_loss(pred_boundary_detail, gt_detail)
    # detail_loss = detail_scale * detail_loss

    # calculate the matte loss
    pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
    matte_l1_loss = F.l1_loss(pred_matte, gt_matte) \
                    + 4.0 * F.l1_loss(pred_boundary_matte, gt_detail) \
                    + F.l1_loss(kornia.filters.sobel(pred_matte), kornia.filters.sobel(gt_detail))
    matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
                               + 4.0 * F.l1_loss(image * pred_boundary_matte, image* gt_detail)
    matte_loss = matte_l1_loss + matte_compositional_loss
    matte_loss = matte_scale * matte_loss

    # matte_l1_loss = disLossSigma(pred_matte, gt_matte, logsigma) + \
    #                 4.0 * disLossSigma(pred_boundary_matte, gt_detail, logsigma)
    # matte_compositional_loss = disLossSigma(image * pred_matte, image * gt_matte, logsigma) \
    #                            + 4.0 * disLossSigma(image * pred_boundary_matte, image * gt_detail, logsigma)
    # matte_loss = matte_l1_loss + matte_compositional_loss
    # matte_loss = matte_scale * matte_loss

    matte_loss_ = criterion_nig(pred_matte, pred_la, pred_alpha, pred_beta, gt_matte, step=epoch, totalStep=args.epoch)
    matte_loss_ = matte_loss_.mean()
    matte_loss += 0 * matte_loss_
    # matte_loss = (matte_loss_ + matte_compositional_loss)
    return matte_loss, matte_compositional_loss


def REITMODNet_Trainer(
        net, image, trimap=None, gt_matte=None, sp=None, instance_map=None, user_map=None,
        mode='modnet', blurer=None, fg=None, bg=None, args=None, epoch=None, cur_step=None, total_step=None):
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
    # show_tensor(show_user*0.3 + 0.7 * instance_map[0][0].unsqueeze(2))

    with autocast():
        # first forward
        input = torch.cat([image, user_map], dim=1)  # , user_map
        pred_la, pred_alpha, pred_beta, pred_matte_s, pred_matte = net(input, False)
        # pred_la2, pred_alpha2, pred_beta2, pred_matte_s2, pred_matte2 = net(input, False)

        # cis_loss = F.l1_loss(pred_la, pred_la2) + \
        #            F.l1_loss(pred_alpha, pred_alpha2) + \
        #            F.l1_loss(pred_beta, pred_beta2) + \
        #            F.l1_loss(pred_matte, pred_matte2)

        #
        # uncertainty
        # with torch.no_grad():
        #     alpha = pred_semantic
        #     alpha = F.interpolate(alpha, scale_factor=4, mode='bilinear')
        #     un = 2 / torch.sum(alpha, dim=1)
        #     n, h, w = un.shape
        #     un[trimap[:, 0, :, :] == 0.5] = 0
        #     un = un.reshape(n, 16, h // 16, 16, w // 16)
        #     un = un.permute([0, 1, 3, 2, 4]).reshape(n, 16 * 16, -1)
        #     patch_un = torch.sum(un, dim=2)
        #     des_index = torch.argsort(patch_un, dim=1, descending=True)
        #     user_map = torch.zeros_like(patch_un)  # (n,256)
        #     for i in range(len(user_map)):
        #         user_map[i, des_index[i, :10]] = 1
        #     user_map = user_map.reshape(n, 1, 16, 16)
        #     user_map = F.interpolate(user_map, (h, w))
        #     user_map -= 1
        #     user_map[user_map == 0] = trimap[user_map == 0]
        #     user_map += 1
        #
        # # user forward
        # user_input = torch.cat([image, user_map], dim=1)
        # user_pred_semantic, user_pred_detail, user_pred_matte = net(user_input, False)

        # cluster
        # target = torch.argmax(cluster, 1)
        # new_target = torch.zeros_like(target) + 255
        # sp_val, sp_num = torch.unique(sp, return_counts=True)
        # '''refine'''
        # for inds in sp_val:
        #     index = sp == inds
        #     if inds == 0:
        #         target[index] = 255
        #         continue
        #     u_labels, hist = torch.unique(target[index], return_counts=True)
        #     new_target[index] = u_labels[torch.argmax(hist)]
        # cluster_loss = F.cross_entropy(cluster, new_target, ignore_index=255)

        # calculate the semantic loss
        # gt_semantic = F.interpolate(gt_matte, scale_factor=1 / 4, mode='bilinear')
        # gt_semantic = blurer(gt_semantic)
        # gt_semantic = kornia.filters.gaussian_blur2d(gt_semantic, (3, 3), (0.8, 0.8))
        # semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))

        # calculate the final loss, backward the loss, and update the model
        matte_loss, matte_compositional_loss = compute_loss(image, gt_matte, trimap, None, None,
                                  pred_matte, pred_la, pred_alpha, pred_beta, epoch, args,
                                  semantic_scale, detail_scale, matte_scale)

        # user_semantic_loss, user_detail_loss, user_matte_loss = compute_loss(image, gt_matte, trimap,
        #                                                                      user_pred_semantic, user_pred_detail,
        #                                                                      user_pred_matte, epoch, args,
        #                                                                      semantic_scale,
        #                                                                      detail_scale, matte_scale)
        # user_loss = 0.5 * user_semantic_loss + user_detail_loss + user_matte_loss
        loss = matte_loss #+ 10 * cis_loss  # + cluster_loss
        loss = loss  # + user_loss

    return {'loss': loss,
            # 'l_s': semantic_loss,
            # 'l_c': cis_loss,
            'l_m': matte_loss,
            # 'u_s': user_semantic_loss,
            'l_c': matte_compositional_loss
            }
