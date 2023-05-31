import random

import cv2
import kornia
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import os
from datasets.augmentations import *
import numpy as np
import datasets.augmentation as A
import torchvision.transforms as T

# from datasets.GFM_dataset2 import MattingTransform
from utils.util import *

#########################
## Data transformer
#########################
CROP_SIZE = [420, 520, 620]
RESIZE_SIZE = 512



# -------------------------------

class BuildLabelDataset_bfd(Dataset):
    """

    """

    def __init__(self, forPath, bgPath, transform=None, augmentation=None, mask=False, trimap=True, mode='no_val',
                 out_size=512):
        """
        @param forPath The path of the foreground files.
        @param bgPath The path of the background files.
        @param mask Whether return the segmentation mask, the mask is a binary image only contains 1 or 0.
        @param pad_size Input size to network.
        """
        self.mode = mode
        self.transform = transform
        # self.transform_gfm = MattingTransform()
        self.augmentation = augmentation
        self.forthPath = forPath
        self.bgPath = bgPath
        self.mask = mask
        self.trimap = trimap
        self.fg_files = os.listdir(forPath)
        if self.mode == 'val':
            self.fg_files += self.fg_files
            self.fg_files += self.fg_files
        self.bg_files = os.listdir(bgPath)
        self.out_size = out_size

        # self.transform_bgm_fg = A.PairCompose([
        #     A.PairRandomAffineAndResize((512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.4, 1),
        #                                 shear=(-5, 5)),
        #     A.PairRandomHorizontalFlip(),
        #     A.PairRandomBoxBlur(0.1, 5),
        #     A.PairRandomSharpen(0.1),
        #     A.PairApplyOnlyAtIndices([0], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
        #     A.PairApply(T.ToTensor())
        # ])
        #
        # self.transform_bgm_bg = T.Compose([
        #     A.RandomAffineAndResize((512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1, 2), shear=(-5, 5)),
        #     T.RandomHorizontalFlip(),
        #     A.RandomBoxBlur(0.1, 5),
        #     A.RandomSharpen(0.1),
        #     T.ColorJitter(0.15, 0.15, 0.15, 0.05),
        #     T.ToTensor()
        # ])

        self.transform_bgm_fg = A.PairCompose([
            A.PairRandomAffineAndResize((out_size, out_size), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.8, 1.2),
                                        shear=(-5, 5)),
            A.PairRandomHorizontalFlip(),
            A.PairRandomBoxBlur(0.4, 5),
            A.PairRandomSharpen(0.3),
            A.PairApplyOnlyAtIndices([0], T.ColorJitter(0.3, 0.15, 0.15, 0.05)),
            # A.PairApplyOnlyAtIndices([1], T.ColorJitter(0.3, 0, 0.15, 0.05)),
            A.PairApply(T.ToTensor())
        ])

        self.transform_bgm_bg = T.Compose([
            A.RandomAffineAndResize((out_size, out_size), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.5, 2),
                                    shear=(-5, 5)),
            T.RandomHorizontalFlip(),
            A.RandomBoxBlur(0.4, 5),
            A.RandomSharpen(0.1),
            T.ColorJitter(0.3, 0.15, 0.15, 0.05),
            T.ToTensor()
        ])

        print("label forground's number:{}".format(len(self.fg_files)))
        print("label background's number:{}".format(len(self.bg_files)))

        # self.training_files = []
        # for fg_name in self.fg_files:
        #     # self.training_files.append([fg_name, '1.png'])
        #     for bg_name in self.bg_files:
        #         self.training_files.append([fg_name, bg_name])

        # print("training data's number:{}".format(len(self.training_files)))

    def __getitem__(self, item):
        # fg_name, bg_name = self.training_files[item]
        fg_name = self.fg_files[item % len(self.fg_files)]
        # bg_index = np.random.randint(0, len(self.bg_files))  # Sampling a bg img.
        bg_index = item % len(self.bg_files)
        bg_im = cv2.imread(self.bgPath + self.bg_files[bg_index])
        # fg_im2 = Image.open(self.forthPath + fg_name)
        fg_im = cv2.imread(self.forthPath + fg_name, cv2.IMREAD_UNCHANGED)

        # show
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(cv2.cvtColor(fg_im[..., :3], cv2.COLOR_BGR2RGB), dtype='uint8'))
        # plt.subplot(1, 2, 2)
        # plt.imshow(fg_im2)
        # plt.savefig('sss.png')

        # fg_im = cropRoiRegion(fg_im)
        if bg_im is None:
            print(self.bg_files[bg_index])
        if len(bg_im.shape) == 2:
            bg_im = cv2.cvtColor(bg_im, cv2.COLOR_GRAY2BGR)
        im_fg, im_alpha = fg_im[..., :3], fg_im[..., 3]
        # im_fg = cv2.cvtColor(im_fg, cv2.COLOR_BGR2RGB)

        if random.random() < 0.5:
            rand_kernel = random.choice([20, 30, 40, 50, 60])
            bg_im = cv2.blur(bg_im, (rand_kernel, rand_kernel))

        # show
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(cv2.cvtColor(im_fg, cv2.COLOR_BGR2RGB), dtype='uint8'))
        # plt.subplot(1, 2, 2)
        # plt.imshow(im_alpha)
        # plt.savefig('sss.png')

        im_fg = Image.fromarray(im_fg, mode='RGB')
        bg_im = Image.fromarray(bg_im, mode='RGB')
        im_alpha = Image.fromarray(im_alpha, mode='L')
        im_fg, im_alpha = self.transform_bgm_fg(im_fg, im_alpha)
        bg_im = self.transform_bgm_bg(bg_im)

        if random.random() < 0.3:
            if random.random() < -1:
                rand_mask = torch.randint_like(im_alpha, 10) * 0.1
                im_alpha = im_alpha * rand_mask.clamp(0.5, 1)
            else:
                im_alpha = im_alpha * random.randint(5, 9) * 0.1
        # shadow
        if random.random() < 0.5:
            aug_shadow = im_alpha.mul(max(0.1, random.random()))
            aug_shadow = T.RandomAffine(degrees=(-5, 5), translate=(0.01, 0.1), scale=(0.95, 1.1), shear=(-5, 5))(
                aug_shadow)
            aug_shadow = kornia.filters.box_blur(aug_shadow.unsqueeze(0), (random.choice(range(20, 40)),) * 2)
            bg_im = bg_im.sub_(aug_shadow[0]).clamp_(0, 1)

        merge_img = im_fg * im_alpha + (1 - im_alpha) * bg_im
        # trimap = get_trimap(im_alpha[0].cpu().numpy())
        trimap = np.zeros(shape=im_alpha[0].shape)
        prior = generateRandomPrior(im_alpha[0].cpu().numpy(), num_fg=[1, 5], num_bg=[1, 5], size=31)
        # prior = get_trimap(im_alpha[0].cpu().numpy())
        # prior[prior == 0] = -1
        # prior[prior == 0.5] = 0

        prior_ = prior.copy()
        prior_[prior_ == -1] = 1
        # prior_trimap = get_trimap(prior_)
        # prior_trimap[prior_trimap == 0.5] = 1
        prior_trimap = prior_

        # if flag == 1:
        #     prior_ = prior.copy()
        #     prior_[prior_ == -1] = 0.5
        #     # prior_trimap = get_trimap(prior_)
        #     # prior_trimap[prior_trimap == 0.5] = 1
        #     prior_trimap = prior_
        # else:
        #     prior_trimap = np.zeros_like(prior)
        #     prior_trimap[im_alpha == 1] = 1
        #     prior_trimap[im_alpha == 0] = 1
        # show
        # plt.subplot(1, 2, 1)
        # plt.imshow(prior * 255, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(prior_trimap * 255, cmap='gray')
        # plt.savefig('sss.png')

        # plt.subplot(1, 3, 1)
        # plt.imshow(
        #     cv2.cvtColor(np.array(merge_img.permute([1, 2, 0]).cpu().numpy() * 255, dtype='uint8'), cv2.COLOR_BGR2RGB))
        # plt.subplot(1, 3, 2)
        # plt.imshow(np.array(im_alpha[0].cpu().numpy() * 255, dtype='uint8'), cmap='gray')
        # plt.subplot(1, 3, 3)
        # plt.imshow(np.array(prior_trimap * 255, dtype='uint8'), cmap='gray')
        # plt.savefig('sss.png')

        # merge_img = im_fg
        label_alpha = im_alpha[0]

        return merge_img, label_alpha, torch.from_numpy(trimap), im_fg, bg_im, torch.from_numpy(
            prior), torch.from_numpy(prior_trimap)

    def __len__(self):
        return max(len(self.bg_files), len(self.fg_files))



#
# class BuildUnLabelDataset(Dataset):
#     def __init__(self, forPath, transform=None, augmentation=None, mask=False, trimap=False, pad_size=(520, 520),
#                  crop_size=(320, 320), mode='train', out_size=512):
#         """
#         @param forPath The path of the foreground files.
#         @param bgPath The path of the background files.
#         @param mask Whether return the segmentation mask, the mask is a binary image only contains 1 or 0.
#         @param pad_size Input size to network.
#         """
#         self.out_size = out_size
#         self.transform = transform
#         self.augmentation = augmentation
#         self.forthPath = forPath
#         self.mask = mask
#         self.trimap = trimap
#         self.pad_size = pad_size
#         self.crop_size = crop_size
#         self.fg_files = os.listdir(forPath)
#         self.training_files = self.fg_files
#         self.transform_gfm = MattingTransform(out_size=self.out_size)
#         self.mode = mode
#
#         print("Unlabeled training data's number:{}".format(len(self.training_files)))
#
#     def __getitem__(self, item):
#         fg_name = self.training_files[item]
#         fg_im = cv2.imread(self.forthPath + fg_name)
#         f_h, f_w, f_c = fg_im.shape
#         assert f_c == 3, "The channel number of fg image requires to equal 3."
#
#         if self.mode == 'train':
#             MAX_SIZE = 520
#             if fg_im.shape[0] > MAX_SIZE or fg_im.shape[1] > MAX_SIZE:
#                 fg_im = scale_img(fg_im, mode='long', size=MAX_SIZE)
#             h, w, c = fg_im.shape
#             rand_ind = random.randint(0, len(CROP_SIZE) - 1)
#             crop_size = CROP_SIZE[rand_ind] if CROP_SIZE[rand_ind] < min(h, w) else min(h, w) - 1
#             resize_size = RESIZE_SIZE
#             ### generate crop centered in transition area randomly
#             fg_im_crop = fg_im[:h - crop_size, :w - crop_size]
#             target = np.where(fg_im_crop > -100)
#             rand_ind = np.random.randint(len(target[0]), size=1)[0]
#             cropx, cropy = target[1][rand_ind], target[0][rand_ind]
#             # # flip the samples randomly
#             flip_flag = True if random.random() < 0.5 else False
#             # generate samples (crop, flip, resize)
#             fg_im = fg_im[cropy:cropy + crop_size, cropx:cropx + crop_size]
#             if flip_flag:
#                 fg_im = cv2.flip(fg_im, 1)
#             fg_im = cv2.resize(fg_im, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
#
#             # Padding to square
#             # if f_h != f_w:
#             #     fg_im = padding_to_square(fg_im)
#             # fg_p_h, fg_p_w, _ = fg_im.shape
#             # fg_im = cv2.resize(fg_im, self.pad_size, interpolation=cv2.INTER_CUBIC)
#             # if self.augmentation is not None:
#             #     fg_im, _ = self.augmentation(fg_im, None)
#             #
#             # if random.random() < 0.5:
#             #     x_r = np.random.randint(0, self.pad_size[0] - self.crop_size[0])
#             #     y_r = np.random.randint(0, self.pad_size[0] - self.crop_size[0])
#             #     fg_im = fg_im[x_r:x_r + self.crop_size[0], y_r:y_r + self.crop_size[0]]
#             # else:
#             #     fg_im = cv2.resize(fg_im, (self.crop_size[0], self.crop_size[0]), interpolation=cv2.INTER_CUBIC)
#             #
#
#         elif self.mode == 'val':
#             fg_im = scale_img(fg_im, mode='long', skip_small=True, size=self.out_size)
#             fg_im = padding_to_square(fg_im)
#
#         if self.transform is not None:
#             fg_im = self.transform(fg_im)
#
#         return fg_im
#
#     def __len__(self):
#         return len(self.training_files)


class BuildOODDataset(Dataset):

    def __init__(self, forPath, bgPath, transform=None, augmentation=None, mask=False, trimap=True,
                 pad_size=(512, 512)):
        """
        @param forPath The path of the foreground files.
        @param bgPath The path of the background files.
        @param mask Whether return the segmentation mask, the mask is a binary image only contains 1 or 0.
        @param pad_size Input size to network.
        """
        self.transform = transform
        self.augmentation = augmentation
        self.forthPath = forPath
        self.bgPath = bgPath
        self.mask = mask
        self.trimap = trimap
        self.pad_size = pad_size
        self.fg_files = os.listdir(forPath)
        self.bg_files = os.listdir(bgPath)
        print("OOD forground's number:{}".format(len(self.fg_files)))
        print("OOD background's number:{}".format(len(self.bg_files)))

    def __getitem__(self, item):
        fg_name = self.fg_files[item]
        bg_index = np.random.randint(0, len(self.bg_files))  # Sampling a bg img.
        bg_im = cv2.imread(self.bgPath + self.bg_files[bg_index])
        fg_im = cv2.imread(self.forthPath + fg_name, cv2.IMREAD_UNCHANGED)
        if len(bg_im.shape) == 2:
            bg_im = cv2.cvtColor(bg_im, cv2.COLOR_GRAY2BGR)

        f_h, f_w, f_c = fg_im.shape
        b_h, b_w, b_c = bg_im.shape

        assert f_c == 4, "The channel number of fg image requires to equal 4."
        assert b_c >= 3, "The channel number of bg image requires to equal 3."

        # Padding to square
        if f_h != f_w:
            fg_im = padding_to_square(fg_im)
        if b_h != b_w:
            bg_im = padding_to_square(bg_im)

        bg_im = cv2.resize(bg_im, (fg_im.shape[0], fg_im.shape[1]), interpolation=cv2.INTER_CUBIC)

        bg_p_h, bg_p_w, _ = bg_im.shape
        fg_p_h, fg_p_w, _ = fg_im.shape

        bg_im = cv2.resize(bg_im, self.pad_size, interpolation=cv2.INTER_CUBIC)
        fg_im = cv2.resize(fg_im, self.pad_size, interpolation=cv2.INTER_CUBIC)
        alpha = fg_im[:, :, 3][:, :, np.newaxis] / 255
        bg_im = (1 - alpha) * bg_im + alpha * fg_im[:, :, :3]
        merge_img = bg_im.astype('uint8')
        label_alpha = np.zeros(shape=alpha.shape, dtype=float)[:, :, 0]

        if self.augmentation is not None:
            merge_img, label_alpha = self.augmentation(merge_img, label_alpha)
            # plt.imshow(merge_img)
            # plt.show()
        x_r = np.random.randint(0, 768 - 512)
        y_r = np.random.randint(0, 768 - 512)
        merge_img = merge_img[x_r:x_r + 512, y_r:y_r + 512]
        label_alpha = label_alpha[x_r:x_r + 512, y_r:y_r + 512]

        trimap = label_alpha + 0.5

        if self.transform is not None:
            merge_img = self.transform(merge_img)

        if self.mask:
            seg_mask = label_alpha.copy()
            seg_mask[seg_mask > 0] = 1
            if self.trimap:
                return merge_img, label_alpha, trimap, seg_mask
            else:
                return merge_img, label_alpha, _, seg_mask
        else:
            if self.trimap:
                # trimap_img = self.get_trimap(label_alpha)
                # cv2.imshow('a', (trimap_img*255).astype('uint8'))
                # cv2.waitKey(300)
                return merge_img, label_alpha, trimap, _
            else:
                return merge_img, label_alpha, _, _

    def __len__(self):
        return len(self.fg_files)


class BuildOODLabelDataset(Dataset):
    """

    """

    def __init__(self, forPath, labelPath, maskPath=None, transform=None, augmentation=None, mask=False, trimap=True,
                 pad_size=(520, 520), mode='val', out_size=512):
        """
        @param forPath The path of the foreground files.
        @param bgPath The path of the background files.
        @param mask Whether return the segmentation mask, the mask is a binary image only contains 1 or 0.
        @param pad_size Input size to network.
        """
        self.mode = mode
        self.transform = transform
        self.augmentation = augmentation
        self.forthPath = forPath
        self.labelPath = labelPath
        self.maskPath = maskPath
        self.mask = mask
        self.trimap = trimap
        self.pad_size = pad_size
        self.fg_files = os.listdir(forPath)
        self.labelPath_files = os.listdir(labelPath)
        self.bg_files = os.listdir(maskPath)
        self.fg_files.sort()
        self.labelPath_files.sort()
        self.out_size = out_size
        print("OOD label dataset's number:{}".format(len(self.fg_files)))

        # self.training_files = []
        # for fg_name in self.fg_files:
        #     # self.training_files.append([fg_name, '1.png'])
        #     for bg_name in self.bg_files:
        #         self.training_files.append([fg_name, bg_name])

        # print("training data's number:{}".format(len(self.training_files)))

    def padding_to_square(self, im):
        h, w, c = im.shape
        top = max(int((w - h) / 2), 0)
        bottom = max(w - top - h, 0)
        left = max(int((h - w) / 2), 0)
        right = max(h - left - w, 0)
        return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    def __getitem__(self, item):
        # fg_name, bg_name = self.training_files[item]
        fg_name = self.fg_files[item]
        label_name = self.labelPath_files[item]
        assert label_name.split('.')[0] == fg_name.split('.')[0], 'name is not match'
        fg_im = cv2.imread(self.forthPath + fg_name)
        label_im = cv2.imread(self.labelPath + label_name)

        if self.mode == 'train':
            bg_index = np.random.randint(0, len(self.bg_files))  # Sampling a bg img.
            bg_name = self.bg_files[bg_index]
            bg_im = cv2.imread(self.maskPath + bg_name)
            h, w, c = fg_im.shape
            bg_im = cv2.resize(bg_im, (w, h))
            if random.random() < 0.5:
                rand_kernel = random.choice([20, 30, 40, 50, 60])
                bg_im = cv2.blur(bg_im, (rand_kernel, rand_kernel))
            # replace randomly background
            if random.random() < 0.7:
                alpha = np.array(label_im[:, :, 0], dtype=float) / 255
                alpha = alpha[:, :, np.newaxis]
                fg_im = fg_im * alpha + (1 - alpha) * bg_im
                fg_im = np.array(fg_im, dtype='uint8')
            f_h, f_w, f_c = fg_im.shape

            # Padding to square
            if f_h != f_w:
                fg_im = padding_to_square(fg_im)
                label_im = padding_to_square(label_im)

            if self.pad_size is not None:
                fg_im = cv2.resize(fg_im, self.pad_size, interpolation=cv2.INTER_LINEAR)
                label_im = cv2.resize(label_im, self.pad_size, interpolation=cv2.INTER_LINEAR)[:, :, 0]
            else:
                fg_im = fg_im
                label_im = label_im[..., 0]

            if self.augmentation is not None:
                merge_img, label_alpha = self.augmentation(fg_im, label_im)
                # plt.imshow(merge_img)
                # plt.show()
                # plt.imshow(label_alpha, cmap='gray')
                # plt.show()
            else:
                merge_img, label_alpha = fg_im, label_im

            if random.random() < 0.5:
                x_r = np.random.randint(0, self.pad_size[0] - self.out_size)
                y_r = np.random.randint(0, self.pad_size[0] - self.out_size)
                merge_img = merge_img[x_r:x_r + self.out_size, y_r:y_r + self.out_size]
                label_alpha = label_alpha[x_r:x_r + self.out_size, y_r:y_r + self.out_size]
            else:
                merge_img = cv2.resize(merge_img, (self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)
                label_alpha = cv2.resize(label_alpha, (self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)

            # plt.imshow(fg_im)
            # plt.show()
        elif self.mode == 'val':
            fg_im = scale_img(fg_im, mode='long', skip_small=True, size=self.out_size)
            label_im = scale_img(label_im, mode='long', skip_small=True, size=self.out_size)
            fg_im = padding_to_square(fg_im)
            label_im = padding_to_square(label_im)
            merge_img, label_alpha = fg_im, label_im[..., 0]

        if self.transform is not None:
            merge_img = self.transform(merge_img)

        if self.mask:
            seg_mask = label_alpha.copy()
            seg_mask[seg_mask > 0] = 1
            if self.trimap:
                return merge_img, label_alpha, get_trimap(label_alpha), seg_mask
            else:
                return merge_img, label_alpha, None, seg_mask
        else:
            if self.trimap:
                # trimap_img = self.get_trimap(label_alpha)
                # cv2.imshow('a', (trimap_img*255).astype('uint8'))
                # cv2.waitKey(300)
                return merge_img, torch.from_numpy(label_alpha) / 255, torch.from_numpy(get_trimap(label_alpha)), 0
            else:
                return merge_img, torch.from_numpy(label_alpha) / 255, 0, 0

    def __len__(self):
        return len(self.fg_files)
        # return 100


class BuildLabelDatasetWGt(Dataset):
    """

    """

    def __init__(self, forPath, labelPath, bgPath=None, mask=False, trimap=True,
                 mode='val', out_size=512):
        """
        @param forPath The path of the foreground files.
        @param bgPath The path of the background files.
        @param mask Whether return the segmentation mask, the mask is a binary image only contains 1 or 0.
        @param pad_size Input size to network.
        """
        self.mode = mode
        self.bg_files = []
        if mode == 'val':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif mode == 'train':
            if bgPath != '' and bgPath is not None:
                self.bg_files = os.listdir(bgPath)
            self.transform_bgm_fg = A.PairCompose([
                A.PairRandomAffineAndResize((out_size, out_size), degrees=(-5, 5), translate=(0.1, 0.1),
                                            scale=(0.8, 1.2),
                                            shear=(-5, 5)),
                A.PairRandomHorizontalFlip(),
                A.PairRandomBoxBlur(0.4, 5),
                A.PairRandomSharpen(0.3),
                A.PairApplyOnlyAtIndices([0], T.ColorJitter(0.3, 0.15, 0.15, 0.05)),
                # A.PairApplyOnlyAtIndices([1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
                A.PairApply(T.ToTensor())
            ])

            self.transform_bgm_bg = T.Compose([
                A.RandomAffineAndResize((out_size, out_size), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.5, 2),
                                        shear=(-5, 5)),
                T.RandomHorizontalFlip(),
                A.RandomBoxBlur(0.4, 5),
                A.RandomSharpen(0.1),
                T.ColorJitter(0.3, 0.15, 0.15, 0.05),
                T.ToTensor()
            ])
        else:
            assert False, 'Not supporting mode: {}' + mode
        self.forthPath = forPath
        self.labelPath = labelPath
        self.bgPath = bgPath
        self.mask = mask
        self.trimap = trimap
        self.fg_files = os.listdir(forPath)
        self.labelPath_files = os.listdir(labelPath)

        # self.trimapPath_files = os.listdir(trimap)
        self.fg_files.sort()
        self.labelPath_files.sort()
        self.out_size = out_size
        print("label dataset's number:{}".format(max(len(self.fg_files), len(self.bg_files))))

        # reweight
        #self.num_samples = len(self.fg_files)
        #self.weights = torch.zeros(size=(self.num_samples,)) + 1.0 / self.num_samples
        #self.loss = torch.zeros_like(self.weights) + 10

    def resetWeight(self, indices, losses):
        self.loss[indices] = losses
        self.weights = self.loss / torch.sum(self.loss)

    def __getitem__(self, item):
        #if self.mode == 'train':
        #    item = torch.multinomial(self.weights, 1, replacement=True)[0]
        # fg_name, bg_name = self.training_files[item]
        fg_index = item % len(self.fg_files)
        fg_name = self.fg_files[fg_index]
        label_name = self.labelPath_files[fg_index]
        assert label_name.split('.')[0] == fg_name.split('.')[0], 'name is not match'
        fg_im = cv2.imread(self.forthPath + fg_name)
        label_im = cv2.imread(self.labelPath + label_name, cv2.IMREAD_UNCHANGED)
        if len(label_im.shape) == 2:
            label_im = label_im
        elif label_im.shape[2] == 4:
            label_im = label_im[..., -1]
        else:
            label_im = label_im[:, :, 0]
        if self.mode == 'train':
            im_fg = Image.fromarray(fg_im, mode='RGB')
            im_alpha = Image.fromarray(label_im, mode='L')
            im_fg, im_alpha = self.transform_bgm_fg(im_fg, im_alpha)

            # replace randomly background
            if len(self.bg_files) > 0 and random.random() < 0.3:
                bg_index = item % len(self.bg_files)
                bg_name = self.bg_files[bg_index]
                bg_im = cv2.imread(self.bgPath + bg_name)
                if len(bg_im.shape) == 2:
                    bg_im = cv2.cvtColor(bg_im, cv2.COLOR_GRAY2BGR)
                if random.random() < 0.5:
                    rand_kernel = random.choice([20, 30, 40, 50, 60])
                    bg_im = cv2.blur(bg_im, (rand_kernel, rand_kernel))
                bg_im = Image.fromarray(bg_im, mode='RGB')
                bg_im = self.transform_bgm_bg(bg_im)

                # random adjust alpha
                # if random.random() < 0.3:
                #     im_alpha = im_alpha * random.randint(5, 9) * 0.1
            else:
                bg_im = im_fg

            # shadow
            if random.random() < 0.5:
                aug_shadow = im_alpha.mul(max(0.1, random.random()))
                aug_shadow = T.RandomAffine(degrees=(-5, 5), translate=(0.01, 0.1), scale=(0.95, 1.1), shear=(-5, 5))(
                    aug_shadow)
                aug_shadow = kornia.filters.box_blur(aug_shadow.unsqueeze(0), (random.choice(range(20, 40)),) * 2)
                bg_im = bg_im.sub_(aug_shadow[0]).clamp_(0, 1)

            merge_img = im_fg * im_alpha + (1 - im_alpha) * bg_im
            prior = generateRandomPrior(im_alpha[0].cpu().numpy(), size=31)
            prior_trimap = prior.copy()
            prior_trimap[prior_trimap == -1] = 1
            label_alpha = im_alpha[0]
            trimap = get_trimap(label_alpha.cpu().numpy())

            # show
            # plt.subplot(1, 7, 1)
            # plt.imshow(np.array(merge_img.permute([1,2,0]).clamp(0,1).detach().cpu().numpy()*255, dtype='uint8'))
            # plt.subplot(1, 7, 2)
            # plt.imshow(label_alpha.detach().cpu().numpy(), cmap='gray')
            # plt.subplot(1, 7, 3)
            # plt.imshow(trimap, cmap='gray')
            # plt.subplot(1, 7, 4)
            # plt.imshow(np.array(im_fg.permute([1,2,0]).detach().cpu().numpy()*255, dtype='uint8'))
            # plt.subplot(1, 7, 5)
            # plt.imshow(np.array(bg_im.permute([1,2,0]).detach().cpu().numpy()*255, dtype='uint8'))
            # plt.subplot(1, 7, 6)
            # plt.imshow(prior, cmap='gray')
            # plt.subplot(1, 7, 7)
            # plt.imshow(prior_trimap, cmap='gray')
            # plt.savefig('sss.png')
            return merge_img, label_alpha.unsqueeze(0), torch.from_numpy(trimap).unsqueeze(0), im_fg, bg_im, \
                   torch.from_numpy(prior).unsqueeze(0), torch.from_numpy(prior_trimap).unsqueeze(0), item
        elif self.mode == 'val':

            fg_im = scale_img(fg_im, mode='long', skip_small=True, size=self.out_size)
            label_im = scale_img(label_im, mode='long', skip_small=True, size=self.out_size)
            fg_im = padding_to_square(fg_im)
            label_im = padding_to_square(label_im)
            merge_img, label_alpha = fg_im, label_im
            # trimap_img = merge_img
            if self.transform is not None:
                merge_img = self.transform(merge_img)
                merge_gt = self.transform(np.array(label_alpha, dtype='uint8'))
            # if self.trimapPath_files is None:
            trimap = get_trimap(merge_gt[0].cpu().numpy())
            # trimap = self.transform(trimap_img)
            # else:
            #     trimap_im = cv2.imread(self.trimapPath_files + label_name)
            #     trimap = Image.fromarray(trimap_im, mode='L')
            return merge_img, merge_gt, trimap

    def __len__(self):
        return max(len(self.bg_files), len(self.fg_files))
        # return 100
