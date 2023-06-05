import random

import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter

"""
Pair transforms are MODs of regular transforms so that it takes in multiple images
and apply exact transforms on all images. This is especially useful when we want the
transforms on a pair of images.

Example:
    img1, img2, ..., imgN = transforms(img1, img2, ..., imgN)
"""


class PairCompose(T.Compose):
    def __call__(self, *x):
        for transform in self.transforms:
            x = transform(*x)
        return x


class PairApply:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *x):
        return [self.transforms(xi) for xi in x]


class PairApplyOnlyAtIndices:
    def __init__(self, indices, transforms):
        self.indices = indices
        self.transforms = transforms

    def __call__(self, *x):
        return [self.transforms(xi) if i in self.indices else xi for i, xi in enumerate(x)]


class PairRandomAffine(T.RandomAffine):
    def __init__(self, degrees, translate=None, scale=None, shear=None, resamples=None, fillcolor=0):
        super().__init__(degrees, translate, scale, shear, Image.NEAREST, fillcolor)
        self.resamples = resamples

    def __call__(self, *x):
        if not len(x):
            return []
        param = self.get_params(self.degrees, self.translate, self.scale, self.shear, x[0].size)
        resamples = self.resamples or [self.resample] * len(x)
        return [F.affine(xi, *param, resamples[i], self.fillcolor) for i, xi in enumerate(x)]


class PairRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __call__(self, *x):
        if torch.rand(1) < self.p:
            x = [F.hflip(xi) for xi in x]
        return x


class RandomBoxBlur:
    def __init__(self, prob, max_radius):
        self.prob = prob
        self.max_radius = max_radius

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            fil = ImageFilter.BoxBlur(random.choice(range(self.max_radius + 1)))
            img = img.filter(fil)
        return img


class PairRandomBoxBlur(RandomBoxBlur):
    def __call__(self, *x):
        if torch.rand(1) < self.prob:
            if torch.rand(1) < 0.5:
                fil = ImageFilter.BoxBlur(random.choice(range(self.max_radius + 1)))
            else:
                fil = ImageFilter.GaussianBlur(random.choice(range(self.max_radius + 1)))
            x = [xi.filter(fil) for xi in x]
        return x


class RandomSharpen:
    def __init__(self, prob):
        self.prob = prob
        self.filter = ImageFilter.SHARPEN

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            img = img.filter(self.filter)
        return img


class PairRandomSharpen(RandomSharpen):
    def __call__(self, *x):
        if torch.rand(1) < self.prob:
            x = [xi.filter(self.filter) for xi in x]
        return x


class PairEraser():
    def __init__(self, s_l=0.02, s_h=0.06, r_1=0.3, r_2=0.6, v_l=0, v_h=255, pixel_level=False):
        self.s_l = s_l
        self.s_h = s_h
        self.r_1 = r_1
        self.r_2 = r_2
        self.v_l = v_l
        self.v_h = v_h
        self.pixel_level = pixel_level

    def __call__(self, *x):
        image = x[0]
        img_h, img_w, img_c = image.shape

        if random.random() > 0.5:
            return [xi for xi in x]

        while True:
            s = np.random.uniform(self.s_l, self.s_h) * img_h * img_w
            r = np.random.uniform(self.r_1, self.r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if self.pixel_level:
            c = np.random.uniform(self.v_l, self.v_h, (h, w, img_c))
        else:
            c = np.random.uniform(self.v_l, self.v_h)

        image[top:top + h, left:left + w, :] = c

        return [xi for xi in x]


class PairRandomAffineAndResize:
    def __init__(self, size, degrees, translate, scale, shear, ratio=(3. / 4., 4. / 3.), resample=Image.BILINEAR,
                 fillcolor=0):
        self.size = size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.ratio = ratio
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, *x):
        if not len(x):
            return []

        w, h = x[0].size
        # scale_factor = max(self.size[1] / w, self.size[0] / h)
        scale_factor = 1

        w_padded = max(w, self.size[1])
        h_padded = max(h, self.size[0])

        pad_h = int(math.ceil((h_padded - h) / 2))
        pad_w = int(math.ceil((w_padded - w) / 2))

        scale = self.scale[0] * scale_factor, self.scale[1] * scale_factor
        if self.translate is not None:
            translate = self.translate[0] * scale_factor, self.translate[1] * scale_factor
        else:
            translate = None
        affine_params = T.RandomAffine.get_params(self.degrees, translate, scale, self.shear, (w, h))

        def transform(img, i):
            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, (pad_w, pad_h))
            if i == -1:
                img = F.affine(img, *affine_params, Image.NEAREST, self.fillcolor)
            else:
                img = F.affine(img, *affine_params, self.resample, self.fillcolor)
            # img = F.center_crop(img, self.size)
            return img

        return [transform(x[i], i) for i in range(len(x))]


class RandomAffineAndResize(PairRandomAffineAndResize):
    def __call__(self, img):
        return super().__call__(img)[0]


# class PairCrop():
#     def __init__(self, out_size):
#         self.out_size = out_size
#
#     def __call__(self, *x):
#         w, h = x[0].size
#         crop_params = T.RandomCrop.get_params(x[0], self.out_size)
#         return [xi for xi in x]


class PairCrop(object):
    def __init__(self, out_size=512, crop_size=[320, 512, 768]):
        self.out_size = out_size
        self.crop_size = crop_size

    def __call__(self, *argv):
        ori = np.array(argv[0])
        h, w, c = ori.shape
        rand_ind = random.randint(0, len(self.crop_size) - 1)
        # crop_size = self.crop_size[rand_ind]
        crop_size = 512
        resize_size = self.out_size
        ### generate crop centered in transition area randomly
        alpha = np.array(argv[-1])
        # alpha_crop = alpha[:h - crop_size, :w - crop_size]
        T_index = (alpha > 0) * (alpha < 255)
        target = np.where(T_index)
        if len(target[0]) == 0:
            target = np.where(alpha > 0)

        rand_ind = np.random.randint(len(target[0]), size=1)[0]
        cropx, cropy = target[1][rand_ind], target[0][rand_ind]
        # generate samples (crop, flip, resize)
        argv_transform = []
        index = 0
        leftx = max(0, int(cropy - crop_size / 2))
        topy = max(0, int(cropx - crop_size / 2))
        for item in argv:
            item = np.array(item)
            item = item[leftx:min(leftx + crop_size, h), topy:min(topy + crop_size, w)]
            if index == 1:
                item = cv2.resize(item, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
            item = cv2.resize(item, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
            argv_transform.append(Image.fromarray(item))
            index += 1

        return argv_transform
