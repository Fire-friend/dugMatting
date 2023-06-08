import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def scale_img(img, mode='short', size=1024, skip_small=True):
    h, w, c = img.shape
    if skip_small and h < size and w < size:
        ratio = 1
    else:
        if mode == 'short':
            ratio = min(h, w) / size
        elif mode == 'long':
            ratio = max(h, w) / size
    n_h = h / ratio
    n_w = w / ratio
    n_h, n_w = max(int(n_h / 32), 1) * 32, max(int(n_w / 32), 1) * 32

    img = cv2.resize(img, (n_w, n_h), interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite('temp.png', img)
    return img


def padding_to_square(im):
    h, w, c = im.shape
    top = max(int((w - h) / 2), 0)
    bottom = max(w - top - h, 0)
    left = max(int((h - w) / 2), 0)
    right = max(h - left - w, 0)
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REFLECT)


def muti_scale_prediction(model, img, input_size=512):
    """
    缩放尺度预测，先初步预测物体大概位置，再将物体裁出来缩放进行预测，与原预测合成
    :param model: 抠图模型，模型返回的最后一个应为透明通道
    :param img: 未缩放的高分辨率原图，要求是正方形图片，可用opencv镜像填充获得正方形
    :param inputs_size: 输入模型的大小
    """
    n, c, h, w = img.shape
    ratio = h / input_size
    # resize_img = cv2.resize(pad_img, (input_size, input_size))
    resize_img = F.interpolate(img, (input_size, input_size), mode='area')
    pred_matte = model(resize_img)[-1]
    matte_arr = pred_matte[0][0].detach().cpu().numpy()
    matte_arr = np.array(matte_arr * 255, dtype='uint8')
    th, binary_matte = cv2.threshold(matte_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    index = np.where(binary_matte == 255)
    x = index[0]
    y = index[1]
    left, right = max(np.min(x) - int(input_size * 0.05), 0), min(np.max(x) + int(input_size * 0.05), input_size)
    top, bottom = max(np.min(y) - int(input_size * 0.05), 0), min(np.max(y) + int(input_size * 0.05), input_size)
    crop_img = img[:, :, int(left * ratio):int(right * ratio), int(top * ratio):int(bottom * ratio)]
    _, _, h_c, w_c = crop_img.shape
    # c_ratio = h_c / input_size
    resize_img_crop = F.interpolate(crop_img, (input_size, input_size), mode='area')
    pred_matte_crop = model(resize_img_crop)[-1]
    resize_pred_matte_crop = F.interpolate(pred_matte_crop, (h_c, w_c), mode='bilinear').detach().cpu().numpy()
    matte = np.zeros(shape=(h, w))
    matte[int(left * ratio):int(right * ratio), int(top * ratio):int(bottom * ratio)] = resize_pred_matte_crop
    return matte


def model_ensemble(model_list, img, mode='mean', center=0.5):
    with torch.no_grad():
        pred_matte_list = []
        for model in model_list:
            pred_matte = model(img)[-1]
            pred_matte_list.append(pred_matte)

        cat_pred_matte = torch.cat(pred_matte_list, dim=0)
        if mode == 'mean':
            merge = torch.mean(cat_pred_matte, dim=0, keepdim=True)
        elif mode == 'max':
            merge = torch.max(cat_pred_matte, dim=0, keepdim=True)[0]
        elif mode == 'min':
            merge = torch.min(cat_pred_matte, dim=0, keepdim=True)[0]
        elif mode == 'confidence':
            confidence = (cat_pred_matte - center) ** 2
            index = torch.argmax(confidence, dim=0)
            one_hot = F.one_hot(index, num_classes=confidence.shape[0])
            one_hot = one_hot.permute([3, 0, 1, 2])
            merge = torch.sum(one_hot * cat_pred_matte, dim=1, keepdim=True)
    return merge


def delSmallRegion(gray_img, area_th=25):
    _, labels_img = cv2.connectedComponents(gray_img)
    labels = np.unique(labels_img)
    max_area = 0

    for cl in labels:
        if cl == 0:
            continue
        index = labels_img == cl
        cur_area = np.sum(index)
        if cur_area < area_th:
            gray_img[index] = 0
            labels_img[index] = 0
        if cur_area > max_area:
            max_area = cur_area

    for cl in labels:
        if cl == 0:
            continue
        index = labels_img == cl
        cur_area = np.sum(index)
        if cur_area < 0.5 * max_area:
            gray_img[index] = 0

    return gray_img


# TODO
def merge_matter_detail_uncertain(matter_pred, detail_pred, uncertain_pred=None, kernel_size=5):
    matter_pred[matter_pred < 100] = 0
    # detail_pred[detail_pred < 100] = 0
    _, labels_detail = cv2.connectedComponents(detail_pred)
    _, labels_matter = cv2.connectedComponents(matter_pred)

    keep_label_detail = []
    for cl_matter in np.unique(labels_matter):
        if cl_matter == 0:
            continue
        index = labels_matter == cl_matter
        if np.sum(labels_detail[index]) != 0:
            keep_label_detail.append(np.unique(labels_detail[index]))
    keep_label_detail = np.unique(np.concatenate(keep_label_detail, axis=0))
    if keep_label_detail[0] == 0:
        keep_label_detail = keep_label_detail[1:]
    mask = np.zeros(shape=labels_detail.shape, dtype='uint8')
    for cl in keep_label_detail:
        mask[labels_detail == cl] = 255
    refine_detail_pred = detail_pred.copy()
    refine_detail_pred[mask != 255] = 0
    refine_detail_pred = delSmallRegion(refine_detail_pred)

    plt.imshow(refine_detail_pred, cmap='gray')
    plt.show()

    refine_matter = matter_pred.copy()
    refine_matter[refine_detail_pred > 100] = refine_detail_pred[[refine_detail_pred > 100]]
    plt.imshow(refine_matter, cmap='gray')
    plt.show()
    print()
    detail_pred[matter_pred == 0] = 0

    matter_pred - detail_pred


def edge_region_refine(img, matte):
    """
    聚类优化抠图 TODO 使用好的聚类算法
    """
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # th, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('binary', binary)
    img = cv2.medianBlur(img, 3)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)  # 转化数据类型
    c = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    k = 10  # 聚类中心个数，一般来说也代表聚类后的图像中的颜色的种类
    ret, label, center = cv2.kmeans(Z, k, None, c, 10, cv2.KMEANS_PP_CENTERS)
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # img9 = res.reshape((img.shape))
    labels_matter = label.reshape((matte.shape))

    # canny = cv2.Canny(img, 20, 200)
    # inverse = 255 - canny
    # _, labels_matter = cv2.connectedComponents(inverse)

    # bg_index = matte < 50
    # labels, labels_num = np.unique(labels_matter[bg_index], return_counts=True)
    # bg_label = labels[np.argmax(labels_num)]
    # matte[labels_matter == bg_label] = 0
    # matte = matte.astype(float) / 255

    labels = np.unique(labels_matter)
    for la in labels:
        index = labels_matter == la
        # maps = np.zeros(shape=matte.shape,dtype='uint8')
        # maps[index] = 255
        # cv2.imshow('aasa', maps)
        # cv2.waitKey(0)
        # _, labels_maps = cv2.connectedComponents(maps)
        # labels_u = np.unique(labels_maps)
        # for la_u in labels_u:
        #     if la_u == 0:
        #         continue
        #     index_ = labels_maps == la_u
        #     num = np.sum(index_)
        #     T_num = np.sum(matte[index_] / 255)
        #     if T_num > 0.5 * num:
        #         matte[index_] = 1
        #     else:
        #         matte[index_] = 0

        num = np.sum(index)
        T_num = np.sum(matte[index] / 255)
        if T_num > 0.5 * num:
            matte[index] = 1
        else:
            matte[index] = 0
        # matte[labels_maps == 0 ] = 0

    # matte = cv2.blur(matte, (3, 3))

    return matte
    # matte = matte.astype(float)
    # matte = matte[:, :, np.newaxis] / 255
    #
    # map = np.zeros(shape=img.shape)
    # map[..., 1] = 255
    # merge = matte * img + (1 - matte) * map
    # merge = np.array(merge, dtype='uint8')
    # print(a)
    # cv2.imshow('merge', merge)
    # cv2.imshow('label', np.array(labels_matter * 20, dtype='uint8'))
    # # cv2.imshow('a', canny)
    # # cv2.imshow('b', inverse)
    # cv2.imshow('matte_refine', matte)
    # cv2.waitKey(0)
