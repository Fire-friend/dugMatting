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
from datasets.data_util import *
# from pytorch_toolbelt.inference import tta
import tta
# from torchstat import stat

from utils.eval import computeAllMatrix
from utils.uncertainty_eva import CalibratedRegression
from utils.util import get_yaml_data, set_yaml_to_args, getPackByNameUtil
from tqdm import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument('--input-path', type=str, help='path of input images',
#                     default='/data/wjw/work/matting_set/data/PPM-100/val/image/')
# parser.add_argument('--gt-path', type=str, help='path of output images',
#                     default='/data/wjw/work/matting_tool_study/test_results/')
# parser.add_argument('--output-path', type=str, help='path of output images',
#                     default='/data/wjw/work/matting_tool_study/test_results/')
# parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet',
#                     default='./checkSave/FBDMv2/AM2K/19/checkpoint/model_best')
parser.add_argument('--model', type=str, help='path of pre-trained MODNet',
                    default='REITMODNet')
parser.add_argument('--gpu', type=str, help='path of pre-trained MODNet',
                    default='0')
parser.add_argument('--save_img', type=bool, help='path of pre-trained MODNet',
                    default=False)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if __name__ == '__main__':
    # define cmd arguments

    # check input arguments
    # if not os.path.exists(args.input_path):
    #     print('Cannot find input path: {0}'.format(args.input_path))
    #     exit()
    # if not os.path.exists(args.output_path):
    #     print('Cannot find output path: {0}'.format(args.output_path))
    #     exit()
    # if not os.path.exists(args.ckpt_path):
    #     print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
    #     exit()

    yamls_dict = get_yaml_data('./config/' + args.model + '_config.yaml')
    set_yaml_to_args(args, yamls_dict)
    # args.save_file = '4'
    args.ckpt_path = './checkSave/{}/{}/{}/checkpoint/model_best'.format(args.model, args.data_set, args.save_file)
    args.save_path = './checkSave/{}/{}/{}/save_imgs_P3MNPGT/'.format(args.model, args.data_set, args.save_file)
    # args.gpu = '1'
    if args.save_img:
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
    # get the dataset dynamically
    dataset = getPackByNameUtil(py_name='datasets.' + args.loader_mode + '_dataset',
                                object_name=args.loader_mode + '_Dataset')
    # get the model dynamically
    model = getPackByNameUtil(py_name='models.' + args.model + '_net',
                              object_name=args.model + '_Net')
    # get the evaluaters dynamically
    try:
        evaluater_iter = getPackByNameUtil(py_name='evaluaters.' + args.model + '_evaluater',
                                           object_name=args.model + '_Evaluater')
        shower_iter = getPackByNameUtil(py_name='evaluaters.' + args.model + '_evaluater',
                                        object_name=args.model + '_Shower')
    except:
        try:
            evaluater_iter = getPackByNameUtil(py_name='evaluaters.' + args.loader_mode + '_evaluater',
                                               object_name=args.loader_mode + '_Evaluater')
            shower_iter = getPackByNameUtil(py_name='evaluaters.' + args.loader_mode + '_evaluater',
                                            object_name=args.loader_mode + '_Shower')
        except:
            evaluater_iter = getPackByNameUtil(py_name='evaluaters.Base_evaluater',
                                               object_name='Base_Evaluater')
            shower_iter = getPackByNameUtil(py_name='evaluaters.Base_evaluater',
                                            object_name='Base_Shower')

    val_dataset = dataset(args, mode='val')
    val_loader = DataLoader(val_dataset,
                            shuffle=True,
                            num_workers=1,
                            batch_size=1,
                            pin_memory=True)
    net = model(args).cuda()
    # load pretrained parameters
    if args.ckpt_path != '' and args.ckpt_path is not None:
        print("loading from {}".format(args.ckpt_path))
        saved_state_dict = torch.load(args.ckpt_path)
        new_params = net.state_dict().copy()
        for name, param in new_params.items():
            if (name in saved_state_dict and param.size() == saved_state_dict[name].size()):
                new_params[name].copy_(saved_state_dict[name])
            elif name[7:] in saved_state_dict and param.size() == saved_state_dict[name[7:]].size():
                new_params[name].copy_(saved_state_dict[name[7:]])
            elif 'module.' + name in saved_state_dict and param.size() == saved_state_dict['module.' + name].size():
                new_params[name].copy_(saved_state_dict['module.' + name])
            else:
                print(name[7:])
        net.load_state_dict(new_params)

    net.eval()

    with torch.no_grad():
        error_sad_sum = 0
        error_mad_sum = 0
        error_mse_sum = 0
        error_grad_sum = 0
        sad_fg_sum = 0
        sad_bg_sum = 0
        sad_tran_sum = 0
        conn_sum = 0
        index = 0
        val_loop = tqdm(enumerate(val_loader), total=len(val_loader))
        val_loop.set_description('val|')
        alea_sum = 0
        epis_sum = 0
        for (i, label_data) in val_loop:
            label_img = label_data[0].cuda().float()
            label_alpha = label_data[1].cuda().float()  # .unsqueeze(1)
            trimap = label_data[2].cuda().float().unsqueeze(1)
            prior = label_data[3].cuda().float()#.unsqueeze(1)
            error_sad, error_mad, error_mse, error_grad, sad_fg, sad_bg, sad_tran, conn, last_un, un, alea_var, input, user_map, last_matte, matte = evaluater_iter(
                net,
                label_img,
                label_alpha,
                trimap,
                prior)
            label_img = input
            alea_sum += torch.mean(un)
            epis_sum += torch.mean(last_un)

            # matte = out[-1]
            if args.save_img:
                _, _, h, w = matte.shape

                # matter -----------------------------
                cv2.imwrite(
                    os.path.join(args.save_path, '{}_matter.png'.format(i + 1)),
                    np.array(matte[0][0].cpu().numpy() * 255, dtype='uint8'))

                # aleatoric --------------------------------------
                un_tensor = un[0]
                un_tensor = (un_tensor - torch.min(un_tensor)) / (torch.max(un_tensor) - torch.min(un_tensor))
                alea = np.array(un_tensor.detach().cpu().numpy() * 255, dtype='uint8')
                alea_heat = cv2.applyColorMap(alea, cv2.COLORMAP_JET)
                # cv2.imwrite(
                #     os.path.join(args.save_path, '{}_alea.png'.format(i + 1)), alea_heat)

                # aleatoric variance -----------------------------
                alea_var_tensor = alea_var[0]
                alea_var_tensor = (alea_var_tensor - torch.min(alea_var_tensor)) / (
                        torch.max(alea_var_tensor) - torch.min(alea_var_tensor))
                alea_var_tensor = np.array(alea_var_tensor.detach().cpu().numpy() * 255, dtype='uint8')
                alea_var_tensor = cv2.applyColorMap(alea_var_tensor, cv2.COLORMAP_JET)
                # cv2.imwrite(
                #     os.path.join(args.save_path, '{}_aleavar.png'.format(i + 1)), alea_var_tensor)

                # epistemic ------------------------------------------------------
                last_un_tensor = last_un[0]
                last_un_tensor = (last_un_tensor - torch.min(last_un_tensor)) / (
                        torch.max(last_un_tensor) - torch.min(last_un_tensor))
                epis__ = np.array(last_un_tensor.detach().cpu().numpy() * 255, dtype='uint8')
                epis_heat = cv2.applyColorMap(epis__, cv2.COLORMAP_JET)
                cv2.imwrite(
                    os.path.join(args.save_path, '{}_epis.png'.format(i + 1)), epis_heat)

                # alea_refine ------------------------------------------------
                epis = (last_un - torch.mean(last_un)) / (torch.std(last_un))
                epis = epis * torch.std(un) + torch.mean(un)
                alea_un = (un[0] - epis[0])
                alea_un[alea_un < 0] = 0
                roi = select_roi(alea_var.unsqueeze(0))
                alea_un[roi[0][0]] = 0
                alea_un = (alea_un - torch.min(alea_un)) / (torch.max(alea_un) - torch.min(alea_un))
                alea = np.array(alea_un.detach().cpu().numpy() * 255, dtype='uint8')
                alea_heat = cv2.applyColorMap(alea, cv2.COLORMAP_JET)
                # cv2.imwrite(
                #     os.path.join(args.save_path, '{}_alea_refine.png'.format(i + 1)), alea_heat)

                # abs err --------------------------------------
                abs_err = torch.abs(label_alpha - last_matte)
                abs_err = (abs_err - torch.min(abs_err)) / (torch.max(abs_err) - torch.min(abs_err))
                abs_err_heat = cv2.applyColorMap(np.array(abs_err[0][0].detach().cpu().numpy() * 255, dtype='uint8'),
                                                 cv2.COLORMAP_JET)
                cv2.imwrite(
                    os.path.join(args.save_path, '{}_abs.png'.format(i + 1)), abs_err_heat)

                # last prediction -----------------------------------------
                label_img = label_img[:, :3] * torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1,
                                                                                        1).cuda() + torch.tensor(
                    [0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()
                cv2.imwrite(
                    os.path.join(args.save_path, '{}_last.png'.format(i + 1)),
                    cv2.cvtColor(
                        np.array((last_matte * label_img + (1 - last_matte) * torch.tensor([0, 1, 0]).view(1, -1, 1,
                                                                                                           1).cuda()).permute(
                            [0, 2, 3, 1])[0].cpu().numpy() * 255,
                                 dtype='uint8'), cv2.COLOR_RGB2BGR))

                # refine prediction ------------------------------------------
                # cv2.imwrite(
                #     os.path.join(args.save_path, '{}.png'.format(i + 1)),
                #     cv2.cvtColor(
                #         np.array((matte * label_img + (1 - matte) * torch.tensor([0, 1, 0]).view(1, -1, 1,
                #                                                                                  1).cuda()).permute(
                #             [0, 2, 3, 1])[0].cpu().numpy() * 255,
                #                  dtype='uint8'), cv2.COLOR_RGB2BGR))

                # input ---------------------------------------------------
                label_img = np.array(label_img.permute([0, 2, 3, 1]).detach().cpu().numpy() * 255, dtype='uint8')[0]
                # cv2.imwrite(
                #     os.path.join(args.save_path, '{}_im.png'.format(i + 1)), cv2.cvtColor(label_img, cv2.COLOR_RGB2BGR)
                # )

                # user map ----------------------------------------------
                user_map = np.array(user_map.detach().cpu().numpy()[0][0])
                user_m = np.zeros_like(label_img)
                user_m[user_map == 1] = [184, 232, 252]
                user_m[user_map == 0.5] = [192, 0, 0]
                user_m[user_map == -1] = [255, 217, 102]

                u_index = user_map != 0
                label_img[u_index, :] = 0 * label_img.astype(float)[u_index, :] + \
                                        1 * user_m.astype(float)[u_index, :]
                save_user = cv2.cvtColor(np.array(label_img, dtype='uint8'), cv2.COLOR_BGR2RGB)


                # cv2.imwrite(
                #     os.path.join(args.save_path, '{}_user.png'.format(i + 1)), save_user
                # )

                # calibration ---------------------------
                def pp_method(x, gt_):
                    return x

                # epis = epis__.astype(float) / 255
                # abs_err = abs_err[0][0].detach().cpu().numpy()
                # index_ = epis > np.mean(epis)
                # epis = epis[index_]
                # abs_err = abs_err[index_]
                # abs_err = (abs_err - np.min(abs_err)) / (np.max(abs_err) - np.min(abs_err))
                # epis = (epis - np.min(epis)) / (np.max(epis) - np.min(epis))
                # calib = CalibratedRegression(epis, abs_err, pp=pp_method, pp_params={'gt_': epis}).fit()
                # plt.style.use('ggplot')
                # fig, ax = plt.subplots()
                # calib.plot_calibration_curve(ax)
                # plt.savefig(os.path.join(args.save_path, '{}_calibration.pdf'.format(i + 1)), bbox_inches='tight',
                #             pad_inches=0.1)
                # plt.savefig(os.path.join(args.save_path, '{}_calibration.svg'.format(i + 1)), bbox_inches='tight',
                #             pad_inches=0.1)
                # plt.close()
                # plt.show(bbox_inches='tight', pad_inches=0.0)

                # cluster = torch.argmax(cluster, dim=1)[0].detach().cpu().numpy()
                # im_arr = np.array(label_img.permute([0, 2, 3, 1]).detach().cpu().numpy()[0] * 255, dtype='uint8')
                # im_arr = cv2.cvtColor(im_arr, cv2.COLOR_RGB2BGR)
                # from skimage import segmentation
                # show_boun = segmentation.mark_boundaries(im_arr, cluster)
                # for cl in np.unique(cluster):
                #     index_ = cluster == cl
                #     im_arr[index_] = np.mean(im_arr[index_], axis=0)
                # cv2.imwrite(os.path.join(args.save_path, '{}_cluster.png'.format(i + 1)),
                #             np.array(im_arr, dtype='uint8'))
                # cv2.imwrite(os.path.join(args.save_path, '{}_cluster2.png'.format(i + 1)),
                #             np.array(show_boun*255, dtype='uint8'))

                # cv2.imwrite(
                #     os.path.join(args.save_path, '{}.png'.format(i + 1)),
                #     np.array(label_alpha[0][0].cpu().numpy() * 255, dtype='uint8'))
            # error_sad, error_mad, error_mse, error_grad, sad_fg, sad_bg, sad_tran = computeAllMatrix(matte,
            #                                                                                          label_alpha,
            #                                                                                          trimap)
            index += error_sad - error_sad + 1
            error_sad_sum += error_sad
            error_mad_sum += error_mad
            error_mse_sum += error_mse
            error_grad_sum += error_grad
            sad_fg_sum += sad_fg
            sad_bg_sum += sad_bg
            sad_tran_sum += sad_tran
            conn_sum += conn

        ave_val_loss = error_mad_sum / index
        ave_error_sad_sum = error_sad_sum / index
        ave_error_mad_sum = error_mad_sum / index
        ave_error_mse_sum = error_mse_sum / index
        ave_error_grad_sum = error_grad_sum / index
        ave_sad_fg_sum = sad_fg_sum / index
        ave_sad_bg_sum = sad_bg_sum / index
        ave_sad_tran_sum = sad_tran_sum / index
        ave_conn_sum = conn_sum / index

        metrix_str = '{:20}\t{:20}\t{:20}\n' \
                     '{:20}\t{:20}\t{:20}\n' \
                     '{:20}\t{:20}\t{:20}\n' \
            .format('Val',
                    'Grad: {:.5f}'.format(ave_error_grad_sum),
                    'Sad: {:.5f}'.format(ave_error_sad_sum),
                    'Mad: {:.5f}'.format(ave_error_mad_sum),
                    'Mse: {:.5f}'.format(ave_error_mse_sum),
                    'Sad_fg: {:.5f}'.format(ave_sad_fg_sum),
                    'Sad_bg: {:.5f}'.format(ave_sad_bg_sum),
                    'Sad_tran: {:.5f}'.format(ave_sad_tran_sum),
                    'CONN: {:.5f}'.format(ave_conn_sum)
                    )
        print(metrix_str)

        print(alea_sum / len(val_loop), epis_sum / len(val_loop))
