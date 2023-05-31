import os
import sys
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from cv2 import resize

from matting_post_tools import edge_region_refine, muti_scale_prediction
from models.bfdnet7 import bfdNet
# from models.FBDM import FBDM as bfdNet
from models.FBDMv2_net_eff import FBDMv2_Net as bfdNet
from models.FBDM_img import FBDM_IMG
from models.gfm import GFM
from models.modnet import MODNet
from models.modnet import MODNet as MODNet_source
from models.u2net import U2NET
# from pytorch_toolbelt.inference import tta
import tta

MAX_SIZE_H = 1600
MAX_SIZE_W = 1600


def gen_bw_from_segmap_e2e(segmap):
    bw = np.argmax(segmap, axis=1)[0]
    bw = bw.astype(np.int64)
    bw[bw == 1] = 255
    return bw.astype(np.uint8)


def get_masked_local_from_global_test(global_result, local_result):
    weighted_global = np.ones(global_result.shape)
    weighted_global[global_result == 255] = 0
    weighted_global[global_result == 0] = 0
    fusion_result = global_result * (1. - weighted_global) / 255 + local_result * weighted_global
    return fusion_result


def gen_trimap_from_segmap_e2e(segmap):
    # segmap = torch.from_numpy(segmap)
    # semantic_probs, trimap = torch.max(F.softmax(segmap, dim=1), dim=1)
    # trimap[semantic_probs < 0.99] = 1
    # trimap = trimap[0].cpu().numpy()

    trimap = np.argmax(segmap, axis=1)[0]
    trimap = trimap.astype(np.int64)
    trimap[trimap == 1] = 128
    trimap[trimap == 2] = 255
    return trimap.astype(np.uint8)


def inference_img_scale(args, model, scale_img):
    pred_list = []
    tensor_img = torch.from_numpy(scale_img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    tensor_img = (tensor_img.cuda() / 255 - 0.5) / 0.5
    input_t = tensor_img
    if args.rosta == 'RIM':
        pred_tt, pred_ft, pred_bt, pred_fusion = model(input_t)
        pred_tt = pred_tt[2].data.cpu().numpy()[0, 0, :, :]
        pred_ft = pred_ft[2].data.cpu().numpy()[0, 0, :, :]
        pred_bt = pred_bt[2].data.cpu().numpy()[0, 0, :, :]
        pred_fusion = pred_fusion.data.cpu().numpy()[0, 0, :, :]
        return pred_tt, pred_ft, pred_bt, pred_fusion
    else:
        # pred_global, pred_local, pred_fusion = model(input_t)
        pred_global, pred_local, pred_fusion = tta.d4_image2mask(model, input_t)
        # pred_fusion = muti_scale_prediction(model, input_t)
        semantic_probs, semantic_index = torch.max(F.softmax(pred_global, dim=1), dim=1)
        semantic_probs = semantic_probs.detach().cpu().numpy()[0]
        if args.rosta == 'TT':
            pred_global = pred_global.data.cpu().numpy()
            pred_global = gen_trimap_from_segmap_e2e(pred_global)
        else:
            pred_global = pred_global.data.cpu().numpy()
            pred_global = gen_bw_from_segmap_e2e(pred_global)
        pred_local = pred_local.data.cpu().numpy()[0, 0, :, :]
        pred_fusion = pred_fusion.data.cpu().numpy()[0, 0, :, :]

        return pred_global, pred_local, pred_fusion, semantic_probs


def inference_img_gfm(args, model, img, option='TT'):
    h, w, c = img.shape
    new_h = min(MAX_SIZE_H, h - (h % 32))
    new_w = min(MAX_SIZE_W, w - (w % 32))

    ## Combine 1/3 glance and 1/2 focus
    # if h > 3000 and w > 3000:
    #     global_ratio = 1 / 3
    #     local_ratio = 1 / 2
    # else:
    #     global_ratio = 1/3
    #     local_ratio = 1/2

    global_ratio = 1
    local_ratio = 1

    resize_h = int(h * global_ratio)
    resize_w = int(w * global_ratio)
    new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
    new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
    scale_img = resize(img, (new_h, new_w))

    pred_glance_1, pred_focus_1, pred_fusion_1, semantic_probs_1 = inference_img_scale(args, model, scale_img)
    pred_glance_1 = resize(pred_glance_1, (h, w), interpolation=cv2.INTER_NEAREST)  # * 255.0
    semantic_probs_1 = resize(semantic_probs_1, (h, w), interpolation=cv2.INTER_LINEAR)  # * 255.0
    resize_h = int(h * local_ratio)
    resize_w = int(w * local_ratio)
    new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
    new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
    scale_img = resize(img, (new_h, new_w))
    pred_glance_2, pred_focus_2, pred_fusion_2, semantic_probs_2 = inference_img_scale(args, model, scale_img)
    pred_focus_2 = resize(pred_focus_2, (h, w), interpolation=cv2.INTER_LINEAR)
    semantic_probs_2 = resize(semantic_probs_2, (h, w), interpolation=cv2.INTER_LINEAR)  # * 255.0
    if option == 'TT':
        pred_fusion = get_masked_local_from_global_test(pred_glance_1, pred_focus_2)
    elif option == 'BT':
        pred_fusion = pred_glance_1 / 255.0 - pred_focus_2
        pred_fusion[pred_fusion < 0] = 0
    else:
        pred_fusion = pred_glance_1 / 255.0 + pred_focus_2
        pred_fusion[pred_fusion > 1] = 1
    return pred_glance_1, pred_focus_2, pred_fusion, semantic_probs_1


def padding_to_square(im):
    h, w, c = im.shape
    top = max(int((w - h) / 2), 0)
    bottom = max(w - top - h, 0)
    left = max(int((h - w) / 2), 0)
    right = max(h - left - w, 0)
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


os.environ['CUDA_VISIBLE_DEVICES'] = '3'

MODEL = 'bfd'  # modnet_refine|modnet_source|gfm|bfd|FBDM_img

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path of input images',
                        default='/data/wjw/work/matting_set/data/PPM-100/val/image/')  # matting_test2021.12.21|user/im_aug
    parser.add_argument('--output-path', type=str, help='path of output images',
                        default='/data/wjw/work/matting_tool_study/test_results/')
    parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet',
                        default='./checkSave/FBDMv2/PPM/19/checkpoint/model_best')  # ./checkSave/bfd/PPM/19/checkSave_bfd/model_10
    parser.add_argument('--backbone', type=str, required=False, default='resnet34',
                        choices=["r34", "r34_2b", "d121", "r101"], help="backbone of GFM")
    parser.add_argument('--rosta', type=str, required=False, default='TT', choices=["TT", "FT", "BT", "RIM"],
                        help="representations of semantic and tarnsition areas")
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.output_path):
        print('Cannot find output path: {0}'.format(args.output_path))
        exit()
    if not os.path.exists(args.ckpt_path):
        print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
        exit()

    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    # im_transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #     ]
    # )
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.55564057, 0.52342629, 0.54125332), (0.34888856, 0.31294156, 0.30838196))  # 归一化
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # create MODNet and load the pre-trained ckpt

    # print(len(train_loader))
    if MODEL == 'modnet_refine':
        modnet = MODNet().cuda()
    elif MODEL == 'modnet_source':
        modnet = MODNet_source(backbone_pretrained=False).cuda()
    elif MODEL == 'gfm':
        modnet = GFM(args)
    elif MODEL == 'bfd':
        # modnet = bfdNet(backbone_pretrained=False, in_w_prior=False)
        modnet = bfdNet(args)
    elif MODEL == 'FBDM_img':
        modnet = FBDM_IMG()
    elif MODEL == 'u2net':
        # modnet = bfdNet(backbone_pretrained=False, stage='refine', in_w_prior=False)
        modnet = U2NET(3, 1)

    # modnet = MODNet(backbone_pretrained=False).cuda()
    modnet = nn.DataParallel(modnet).cuda()
    # modnet = nn.DataParallel(modnet).cuda()
    # modnet.load_state_dict(torch.load(args.ckpt_path))
    print("loading from {}".format(args.ckpt_path))
    map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
    saved_state_dict = torch.load(args.ckpt_path, map_location='cuda:0')
    new_params = modnet.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
        else:
            print(name)
    modnet.load_state_dict(new_params)

    # map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
    # saved_state_dict = torch.load(args.ckpt_path, map_location='cuda:0')
    # new_params = modnet.state_dict().copy()
    # for name, param in saved_state_dict.items():
    #     if name in new_params and new_params[name].size() == param.size():
    #         new_params[name].copy_(saved_state_dict[name])
    #     else:
    #         print(name)
    # modnet.load_state_dict(new_params)

    modnet.eval()

    # inference images
    im_names = os.listdir(args.input_path)
    for im_name in im_names:
        print('Process image: {0}'.format(im_name))

        # read image
        # im = Image.open(os.path.join(args.input_path, im_name))
        im = cv2.imread(os.path.join(args.input_path, im_name), cv2.IMREAD_UNCHANGED)

        im = padding_to_square(im)
        # im = cv2.resize(im, (ref_size, ref_size), interpolation=cv2.INTER_CUBIC)
        if im.shape[2] == 4:
            al = im[:, :, 3]
            im = im[:, :, :3]
            im[al == 0, :] = 255
        source = im.copy()
        # cv2.imshow('ss', source)
        # cv2.waitKey(50)

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]
        im_arr = im.copy()

        # convert image to PyTorch tensor
        # im = Image.fromarray(im)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape

        if True:  # min(im_h, im_w) > ref_size:  # max(im_h, im_w) < ref_size or #True:  #True:  #
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        # im_rw = im_w - im_w % 32
        # im_rh = im_h - im_h % 32

        im_re = F.interpolate(im, size=(im_rh, im_rw), mode='bicubic')

        # inference
        if MODEL == 'modnet_source':
            with torch.no_grad():
                pred_semantic, pred_detail, pred_matte = modnet(im_re.cuda(), False)
        elif MODEL == 'modnet_refine':
            pred_semantic, pred_semantic2, pred_semantic3, pred_detail, pred_matte = modnet(im_re.cuda(), False)
        elif MODEL == 'gfm':
            # pred_semantic, pred_detail, pred_matte = modnet(im.cuda())
            pred_semantic, pred_detail, pred_matte, semantic_probs = inference_img_gfm(args, modnet, im_arr)
            # refine_matte = edge_region_refine(im_arr, pred_matte.copy() * 255)
            refine_matte = pred_matte
        elif MODEL == 'FBDM_img':
            # pred_semantic, pred_detail, pred_matte = modnet(im.cuda())
            _, pred_matte = modnet(im_re.cuda())
            # refine_matte = edge_region_refine(im_arr, pred_matte.copy() * 255)
            refine_matte = pred_matte
        elif MODEL == 'u2net':
            with torch.no_grad():
                d0, d1, d2, d3, d4, d5, d6 = modnet(im_re.cuda(), mode='val')
                # res = tta.d4_image2mask(modnet, im_re.cuda())
                # (d0, e0), (d1, e1), (d2, e2), (d3, e3), (d4, e4), (d5, e5), (d6, e6) = res
        elif MODEL == 'bfd':
            prior = torch.zeros_like(im_re)[:, :1]
            with torch.no_grad():
                # bg_out, fg_out, detail = modnet(im_re.cuda())
                # bg_out, fg_out, err_out, fusion_bg, fusion_fg, fusion_err, matte_out = modnet(im_re.cuda())
                # prior, fg_out, bg_out, fgr_out, bgr_out, \
                # fusion_fg, fusion_bg, \
                # fusion_fgr, fusion_bgr, \
                # fg_refine, bg_refine, fgr_refine, bgr_refine, \
                # matte_refine = modnet(im_re.cuda(), prior=prior.cuda())

                fg_out_list, bg_out_list, detail_out_fg_list, detail_out_bg_list, prior_fg, prior_bg, out = modnet(
                    im_re.cuda(), prior=None, mode='val')

            # b_g_map = torch.abs(fusion_bg + fusion_fg - 1)
            # fgr_refine_im = torch.abs(fgr_refine + im_re.cuda()).clamp(0, 1)
            # bgr_refine_im = torch.abs(bgr_refine + im_re.cuda()).clamp(0, 1)
            # matte = 0.5*(1-bg_out[-1]) + 0.5*fg_out[-1]
            # matte[matte < 0.4] = 0
            # matte[matte > 0.4] = 1
        else:
            assert False, 'the mode is not supprot'

        # resize and save matte
        if MODEL == 'gfm':
            refine_matte = cv2.resize(refine_matte, (im_w, im_h))
            matte = cv2.resize(pred_matte, (im_w, im_h))
            semantic_matte = cv2.resize(pred_semantic / 255, (im_w, im_h))
            detail_matte = cv2.resize(pred_detail, (im_w, im_h))
            semantic_probs = cv2.resize(semantic_probs, (im_w, im_h))
        elif MODEL == 'FBDM_img':
            refine_matte = cv2.resize(refine_matte, (im_w, im_h))
            matte = cv2.resize(pred_matte, (im_w, im_h))
            semantic_matte = cv2.resize(pred_semantic / 255, (im_w, im_h))
            detail_matte = cv2.resize(pred_detail, (im_w, im_h))
            semantic_probs = cv2.resize(semantic_probs, (im_w, im_h))
        elif MODEL == 'u2net':
            matte = d0[0][0].detach().cpu().numpy()
            matte1 = d2[0][0].detach().cpu().numpy()
            matte2 = d3[0][0].detach().cpu().numpy()
            matte_refine_name = im_name.split('.')[0] + 'matte.png'  # '{}__'.format(e0.item())+
            matte_refine_name1 = im_name.split('.')[0] + 'matte1.png'
            matte_refine_name2 = im_name.split('.')[0] + 'matte2.png'
            merge_name = im_name.split('.')[0] + 'merge.png'
            merge_name1 = im_name.split('.')[0] + 'merge1.png'
            merge_name2 = im_name.split('.')[0] + 'merge2.png'
            source_name = im_name.split('.')[0] + 'source.png'

            source_ = cv2.resize(source, (matte.shape[0], matte.shape[1])).astype(float) / 255

            bg = np.zeros(shape=source_.shape)
            bg[:, :, 1] = 1

            merge = source_ * matte[:, :, np.newaxis] + bg * (1 - matte[:, :, np.newaxis])
            merge1 = source_ * matte1[:, :, np.newaxis] + bg * (1 - matte1[:, :, np.newaxis])
            merge2 = source_ * matte2[:, :, np.newaxis] + bg * (1 - matte2[:, :, np.newaxis])
            cv2.imwrite(os.path.join(args.output_path, merge_name), np.array(merge * 255, dtype='uint8'))
            cv2.imwrite(os.path.join(args.output_path, matte_refine_name), np.array(matte * 255, dtype='uint8'))
            cv2.imwrite(os.path.join(args.output_path, source_name), np.array(source_ * 255, dtype='uint8'))
            # cv2.imwrite(os.path.join(args.output_path, merge_name1), np.array(merge1 * 255, dtype='uint8'))
            # cv2.imwrite(os.path.join(args.output_path, matte_refine_name1), np.array(matte1 * 255, dtype='uint8'))
            # cv2.imwrite(os.path.join(args.output_path, merge_name2), np.array(merge2 * 255, dtype='uint8'))
            # cv2.imwrite(os.path.join(args.output_path, matte_refine_name2), np.array(matte2 * 255, dtype='uint8'))

            continue


        elif MODEL == 'bfd':
            im_h, im_w = ref_size, ref_size
            prior = F.interpolate(prior, size=(im_h, im_w), mode='area').clamp(0, 1)
            # detail_fusion = F.interpolate(fusion_detail, size=(im_h, im_w), mode='area').clamp(0, 1)
            # fg_fusion = F.interpolate(fusion_fg, size=(im_h, im_w), mode='area').clamp(0, 1)
            # bg_fusion = F.interpolate(fusion_bg, size=(im_h, im_w), mode='area').clamp(0, 1)
            # detail_refine = F.interpolate(detail_refine, size=(im_h, im_w), mode='area').clamp(0, 1)
            # fg_refine = F.interpolate(fg_refine, size=(im_h, im_w), mode='area').clamp(0, 1)
            # bg_refine = F.interpolate(bg_refine, size=(im_h, im_w), mode='area').clamp(0, 1)
            # fgr_refine_im = F.interpolate(fgr_refine_im, size=(im_h, im_w), mode='area').permute([0, 2, 3, 1])
            # bgr_refine_im = F.interpolate(bgr_refine_im, size=(im_h, im_w), mode='area').permute([0, 2, 3, 1])
            matte_refine = F.interpolate(out, size=(im_h, im_w), mode='area')

            prior_fg = torch.sigmoid(prior_fg)[0][0].data.cpu().numpy()
            prior_bg = torch.sigmoid(prior_bg)[0][0].data.cpu().numpy()
            # fg_fusion = fg_fusion[0][0].data.cpu().numpy()
            # bg_fusion = bg_fusion[0][0].data.cpu().numpy()
            # matte = (matte - np.min(matte)) / (np.max(matte) - np.min(matte))
            # detail_refine = detail_refine[0][0].data.cpu().numpy()
            # fg_refine = fg_refine[0][0].data.cpu().numpy()
            # bg_refine = bg_refine[0][0].data.cpu().numpy()
            # fgr_refine_im = fgr_refine_im[0].data.cpu().numpy()
            # bgr_refine_im = bgr_refine_im[0].data.cpu().numpy()
            matte_refine = matte_refine[0][0].data.cpu().numpy()
            # err = np.abs(fg_fusion + bg_fusion - 1)

            prior_fg_name = im_name.split('.')[0] + 'prior_fg.png'
            prior_bg_name = im_name.split('.')[0] + 'prior_bg.png'
            # detail_fusion_name = im_name.split('.')[0] + 'detail_fusion.png'
            # fg_fusion_name = im_name.split('.')[0] + 'fg_fusion.png'
            # bg_fusion_name = im_name.split('.')[0] + 'bg_fusion.png'
            # err_fusion_name = im_name.split('.')[0] + 'err_fusion.png'
            # fg_refine_name = im_name.split('.')[0] + 'fg_refine.png'
            # bg_refine_name = im_name.split('.')[0] + 'bg_refine.png'
            # fgr_refine_name = im_name.split('.')[0] + 'fgr_refine.png'
            # bgr_refine_name = im_name.split('.')[0] + 'bgr_refine.png'
            matte_refine_name = im_name.split('.')[0] + 'matte.png'
            merge_name = im_name.split('.')[0] + 'merge.png'

            source_ = cv2.resize(source, (matte_refine.shape[0], matte_refine.shape[1])).astype(float) / 255

            bg = np.zeros(shape=source_.shape)
            bg[:, :, 1] = 1

            # im_np = im_re.permute([0,2,3,1])[0].detach().cpu().numpy()
            # matte_refine = np.clip((im_np - bgr_refine_im) / (fgr_refine_im - bgr_refine_im), 0, 1)[...,0]
            # merge = fgr_refine_im * matte_refine[:, :, np.newaxis] + bg * (1 - matte_refine[:, :, np.newaxis])

            merge = source_ * matte_refine[:, :, np.newaxis] + bg * (1 - matte_refine[:, :, np.newaxis])

            # cv2.imwrite(os.path.join(args.output_path, prior_name),
            #             np.array(prior * 255, dtype='uint8'))
            # plt.imshow(prior_fg, cmap='gray')
            # plt.savefig(os.path.join(args.output_path, prior_fg_name))
            # plt.imshow(prior_bg, cmap='gray')
            # plt.savefig(os.path.join(args.output_path, prior_bg_name))
            cv2.imwrite(os.path.join(args.output_path, prior_fg_name), np.array(prior_fg * 255, dtype='uint8'))
            cv2.imwrite(os.path.join(args.output_path, prior_bg_name), np.array(prior_bg * 255, dtype='uint8'))

            # cv2.imwrite(os.path.join(args.output_path, fg_fusion_name), np.array(fg_fusion * 255, dtype='uint8'))
            # cv2.imwrite(os.path.join(args.output_path, bg_fusion_name), np.array(bg_fusion * 255, dtype='uint8'))
            # heat_img = cv2.applyColorMap(np.array(err * 255, dtype='uint8'),cv2.COLORMAP_JET)  # 此处的三通道热力图是cv2使用GBR排列
            # cv2.imwrite(os.path.join(args.output_path, err_fusion_name), heat_img)
            # cv2.imwrite(os.path.join(args.output_path, fg_refine_name), np.array(fg_refine * 255, dtype='uint8'))
            # cv2.imwrite(os.path.join(args.output_path, bg_refine_name), np.array(bg_refine * 255, dtype='uint8'))
            # cv2.imwrite(os.path.join(args.output_path, fgr_refine_name), np.array(fgr_refine_im * 255, dtype='uint8'))
            # cv2.imwrite(os.path.join(args.output_path, bgr_refine_name), np.array(bgr_refine_im * 255, dtype='uint8'))
            cv2.imwrite(os.path.join(args.output_path, merge_name), np.array(merge * 255, dtype='uint8'))
            cv2.imwrite(os.path.join(args.output_path, matte_refine_name), np.array(matte_refine * 255, dtype='uint8'))
            continue
        else:
            semantic_probs = None
            matte = F.interpolate(pred_matte, size=(im_h, im_w), mode='area')
            matte = matte[0][0].data.cpu().numpy()
            # with torch.no_grad():
            #     refine_matte_cluster = edge_region_refine(im_arr, matte.copy() * 255)
            #     refine_matte_scale = muti_scale_prediction(modnet, im.cuda())
            #     _, _, pred_matte_tta = tta.d4_image2mask(modnet, im_re.cuda())
            #     matte_tta = F.interpolate(pred_matte_tta, size=(im_h, im_w), mode='area')
            #     refine_matte_tta = matte_tta[0][0].data.cpu().numpy()
            #     # refine_matte_tta = matte

            semantic_matte = F.interpolate(pred_semantic, size=(im_h, im_w), mode='area')
            semantic_matte = semantic_matte[0][0].data.cpu().numpy()

            detail_matte = F.interpolate(pred_detail, size=(im_h, im_w), mode='area')
            detail_matte = detail_matte[0][0].data.cpu().numpy()

        # _, binary = cv2.threshold((detail_matte * 255).astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 找到最大区域并填充
        # area = []
        # for j in range(len(contours)):
        #     area.append(cv2.contourArea(contours[j]))
        # if len(area) != 0:
        #     max_idx = np.argmax(area)
        #     max_area = cv2.contourArea(contours[max_idx])
        #     binary = cv2.drawContours(binary, contours, max_idx, 255, cv2.FILLED)
        #     bi = im_name.split('.')[0] + 'bi.png'
        #
        #     for k in range(len(contours)):
        #         if k != max_idx:
        #             cv2.fillPoly(binary, [contours[k]], 0)

        matte_name = im_name.split('.')[0] + '.png'
        ims_name = im_name.split('.')[0] + 'im.png'
        refine_ims_name_tta = im_name.split('.')[0] + 'tta.png'
        refine_ims_name_scale = im_name.split('.')[0] + 'scale.png'
        refine_ims_name_cluster = im_name.split('.')[0] + 'cluster.png'
        se_name = im_name.split('.')[0] + 'se.png'
        de = im_name.split('.')[0] + 'de.png'
        heat_name = im_name.split('.')[0] + 'heat.png'

        # Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(args.output_path, matte_name))
        # Image.fromarray(((semantic_matte * 255).astype('uint8')), mode='L').save(
        #     os.path.join(args.output_path, se_name))
        # Image.fromarray(((detail_matte * 255).astype('uint8')), mode='L').save(os.path.join(args.output_path, de))
        # Image.fromarray(binary, mode='L').save(os.path.join(args.output_path, bi))

        # ims = np.concatenate([source, (matte * 255).astype('uint8')[:, :, np.newaxis]], axis=2)
        # refine_ims = np.concatenate([source, (refine_matte * 255).astype('uint8')[:, :, np.newaxis]], axis=2)
        bg = np.zeros(shape=source.shape)
        bg[:, :, 1] = 255

        # matte = (cv2.medianBlur((matte * 255).astype('uint8'), 5)).astype(float) / 255

        # matte[matte < 0.4] = 0
        # matte[matte > 0.7] = 1
        # refine_matte_cluster[refine_matte_cluster < 0.4] = 0
        # # refine_matte_cluster[refine_matte_cluster > 0.7] = 1
        # refine_matte_tta[refine_matte_tta < 0.4] = 0
        # # refine_matte_tta[refine_matte_tta > 0.7] = 1
        # refine_matte_scale[refine_matte_scale < 0.4] = 0
        # # refine_matte_scale[refine_matte_scale > 0.7] = 1

        merge = source * matte[:, :, np.newaxis] + bg * (1 - matte[:, :, np.newaxis])
        # refine_merge_cluster = source * refine_matte_cluster[:, :, np.newaxis] + bg * (
        #         1 - refine_matte_cluster[:, :, np.newaxis])
        # refine_merge_tta = source * refine_matte_tta[:, :, np.newaxis] + bg * (
        #         1 - refine_matte_tta[:, :, np.newaxis])
        # refine_merge_scale = source * refine_matte_scale[:, :, np.newaxis] + bg * (
        #         1 - refine_matte_scale[:, :, np.newaxis])

        # cv2.imwrite(os.path.join(args.output_path, ims_name), ims, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # merge = cv2.medianBlur(merge.astype('uint8'), 3)
        cv2.imwrite(os.path.join(args.output_path, ims_name), merge)
        cv2.imwrite(os.path.join(args.output_path, de), np.array(detail_matte * 255, dtype='uint8'))
        cv2.imwrite(os.path.join(args.output_path, se_name), np.array(semantic_matte * 255, dtype='uint8'))
        # cv2.imwrite(os.path.join(args.output_path, refine_ims_name_tta), refine_merge_tta)
        # cv2.imwrite(os.path.join(args.output_path, refine_ims_name_cluster), refine_merge_cluster)
        # cv2.imwrite(os.path.join(args.output_path, refine_ims_name_scale), refine_merge_scale)
        if semantic_probs is not None:
            heat_img = cv2.applyColorMap(np.array(semantic_probs * 255, dtype='uint8'),
                                         cv2.COLORMAP_JET)  # 此处的三通道热力图是cv2使用GBR排列
            cv2.imwrite(os.path.join(args.output_path, heat_name), heat_img)
