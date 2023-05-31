import torch

from utils.eval import computeAllMatrix


def Base_Evaluater(net, input, gt, trimap):
    input = torch.cat([input, torch.zeros_like(gt)], dim=1)
    out = net(input)
    matte = out[-1]
    error_sad, error_mad, error_mse, error_grad, sad_fg, sad_bg, sad_tran = computeAllMatrix(matte,
                                                                                             gt,
                                                                                             trimap)
    return error_sad, error_mad, error_mse, error_grad, sad_fg, sad_bg, sad_tran, out, input

def Base_Shower(net, input, gt):
    input = torch.cat([input, torch.zeros_like(gt)], dim=1)
    out = net(input)
    matte = out[-1]

    return matte, input