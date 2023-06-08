import importlib
import os
import random
import shutil
from typing import Tuple

import torch
import numpy as np
import cv2
import yaml
from PIL import Image

from utils.util import show_tensor


def getInstanceMap(alpha: torch.tensor):
    """
    alpha [0,1]
    """
    target = torch.where(alpha == 1)
    # centerX = torch.mean(target[0].float())
    # centerY = torch.mean(target[1].float())
    # pathc_size = 20
    # X_index = (target[0]> centerX-pathc_size) * (target[0] < centerX+pathc_size)
    # newTarget = (target[0][X_index], target[1][X_index])
    # Y_index = (newTarget[1] > centerY - pathc_size) * (newTarget[1] < centerY + pathc_size)
    # newTarget = (newTarget[0][Y_index], newTarget[1][Y_index])

    newTarget = target
    if len(newTarget[0]) < 2:
        newTarget = np.where(alpha > -1)

    num = random.randint(1, 5)
    point = []
    # print('new', newTarget)
    for i in range(num):
        rand_ind = np.random.randint(len(newTarget[0]), size=1)[0]
        x, y = newTarget[0][rand_ind], newTarget[1][rand_ind]
        point.append([x, y])
    point = torch.tensor(point)

    instanceMap = torch.zeros_like(alpha)
    h, w = instanceMap.shape
    xnums = torch.arange(0, h)
    xnums = xnums.unsqueeze(1)
    xmap = instanceMap + xnums
    ynums = torch.arange(0, w)
    ynums = ynums.unsqueeze(0)
    ymap = instanceMap + ynums
    xmap = xmap.unsqueeze(0)
    ymap = ymap.unsqueeze(0)
    instanceMap = ((xmap - point[:, 0:1].unsqueeze(2)) ** 2 + (ymap - point[:, 1:2].unsqueeze(2)) ** 2) ** (1 / 3)
    instanceMap = torch.min(instanceMap, dim=0)[0]
    instanceMap = (instanceMap - torch.min(instanceMap)) / (torch.max(instanceMap) - torch.min(instanceMap))
    # show_tensor(instanceMap, mode='gray')
    return instanceMap


from sklearn.cluster import KMeans


def generateRandomPriorDIM(alpha, r=5, mode='circle', val=False):
    prior = np.zeros(shape=alpha.shape)
    h, w = prior.shape
    # nums_fg = max(0,random.randint(-3, 3))
    if val:
        nums_fg = 10
        nums_bg = 5
        nums_t = 5
    else:
        if random.random() < 0.5:
            nums_fg = np.random.geometric(1 / 6)
        else:
            nums_fg = 0
        if random.random() < 0.5:
            nums_bg = np.random.geometric(1 / 6)
        else:
            nums_bg = 0
        if random.random() < 0.5:
            nums_t = np.random.geometric(1 / 6)
        else:
            nums_t = 0
        #
        # nums_fg = max(0, random.randint(-3, 6))
        # nums_bg = max(0, random.randint(-3, 6))
        # nums_t = max(0, random.randint(-3, 6))

    # nums = 30
    fg_index = np.where(alpha == 1)  # [pad + 1:-pad - 1, pad + 1:-pad - 1]
    if len(fg_index[0]) > 0:
        if val:
            estimator = KMeans(n_clusters=nums_fg)  # 构造聚类器
            fg_points = np.array(fg_index).T
            estimator.fit(fg_points)
            centroids = np.array(estimator.cluster_centers_, dtype=int)  # 获取聚类中心
            dist = np.sqrt(np.sum((fg_points[np.newaxis, :, :] - centroids[:, np.newaxis, :]) ** 2, axis=2))
            index = np.argmin(dist, axis=1)
            for l in index:
                prior[max(fg_points[l][0] - r, 0):min(fg_points[l][0] + r, h),
                max(fg_points[l][1] - r, 0):min(fg_points[l][1] + r, w)] = 1
        else:
            for i in range(nums_fg):
                idx = random.randint(0, len(fg_index[0]) - 1)
                if mode == 'rect':
                    prior[max(fg_index[0][idx] - r, 0):min(fg_index[0][idx] + r, h),
                    max(fg_index[1][idx] - r, 0):min(fg_index[1][idx] + r, w)] = 1
                elif mode == 'circle':
                    cv2.circle(prior, (fg_index[1][idx], fg_index[0][idx]), r, 1, -1)

    # nums = 30
    bg_index = np.where(alpha == 0)  # [pad + 1:-pad - 1, pad + 1:-pad - 1]
    if len(bg_index[0]) > 0:
        if val:
            estimator = KMeans(n_clusters=nums_bg)  # 构造聚类器
            bg_points = np.array(bg_index).T
            estimator.fit(bg_points)
            centroids = np.array(estimator.cluster_centers_, dtype=int)  # 获取聚类中心
            dist = np.sqrt(np.sum((bg_points[np.newaxis, :, :] - centroids[:, np.newaxis, :]) ** 2, axis=2))
            index = np.argmin(dist, axis=1)
            for l in index:
                prior[max(bg_points[l][0] - r, 0):min(bg_points[l][0] + r, h),
                max(bg_points[l][1] - r, 0):min(bg_points[l][1] + r, w)] = -1
        else:
            for i in range(nums_bg):
                idx = random.randint(0, len(bg_index[0]) - 1)
                if mode == 'rect':
                    prior[max(bg_index[0][idx] - r, 0):min(bg_index[0][idx] + r, h),
                    max(bg_index[1][idx] - r, 0):min(bg_index[1][idx] + r, w)] = -1
                elif mode == 'circle':
                    cv2.circle(prior, (bg_index[1][idx], bg_index[0][idx]), r, -1, -1)

    # nums = 30
    t_index = np.where((alpha > 0) * (alpha < 1))  # [pad + 1:-pad - 1, pad + 1:-pad - 1]
    if len(t_index[0]) > 0:
        if val:
            estimator = KMeans(n_clusters=nums_t)  # 构造聚类器
            t_points = np.array(t_index).T
            estimator.fit(t_points)
            centroids = np.array(estimator.cluster_centers_, dtype=int)  # 获取聚类中心
            dist = np.sqrt(np.sum((t_points[np.newaxis, :, :] - centroids[:, np.newaxis, :]) ** 2, axis=2))
            index = np.argmin(dist, axis=1)
            for l in index:
                prior[max(t_points[l][0] - r, 0):min(t_points[l][0] + r, h),
                max(t_points[l][1] - r, 0):min(t_points[l][1] + r, w)] = 0.5
        else:
            for i in range(nums_t):
                idx = random.randint(0, len(t_index[0]) - 1)
                if mode == 'rect':
                    prior[max(t_index[0][idx] - r, 0):min(t_index[0][idx] + r, h),
                    max(t_index[1][idx] - r, 0):min(t_index[1][idx] + r, w)] = 0.5
                elif mode == 'circle':
                    cv2.circle(prior, (t_index[1][idx], t_index[0][idx]), r, 0.5, -1)

    # if np.sum(prior == 1) == 0:
    #     prior_index = np.where(alpha == 1)  # [pad + 1:-pad - 1, pad + 1:-pad - 1]
    #     nums = np.random.geometric(1 / 6)
    #     if len(prior_index[0]) > 0:
    #         for i in range(nums):
    #             idx = random.randint(0, len(prior_index[0]) - 1)
    #             if mode == 'rect':
    #                 prior[max(prior_index[0][idx] - r, 0):min(prior_index[0][idx] + r, h),
    #                 max(prior_index[1][idx] - r, 0):min(prior_index[1][idx] + r, w)] = 1
    #             else:
    #                 cv2.circle(prior, (prior_index[1][idx], prior_index[0][idx]), r, 1, -1)
    return prior
