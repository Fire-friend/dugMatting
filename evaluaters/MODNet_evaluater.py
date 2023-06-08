import torch

from utils.eval import computeAllMatrix
import torch.nn.functional as F

def MODNet_Evaluater(net, input, gt, trimap, fusion=False, interac=2):
    out = net(input, True)
    matte = out[-1]
    if matte.shape[2] != gt.shape[2]:
        matte = F.interpolate(matte, (gt.shape[2], gt.shape[3]), mode='bilinear')
    error_sad, error_mad, error_mse, error_grad, sad_fg, sad_bg, sad_tran, conn = computeAllMatrix(matte,
                                                                                                   gt,
                                                                                             trimap)
    return error_sad, error_mad, error_mse, error_grad, sad_fg, sad_bg, sad_tran, conn, out, input

def MODNet_Shower(net, input, gt):
    input = torch.cat([input, torch.zeros_like(gt)], dim=1)
    out = net(input)
    matte = out[-1]

    return matte, input