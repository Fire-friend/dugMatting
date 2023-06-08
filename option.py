import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for the training purpose.')
    # backbone: the backbone of GFM, we provide four backbones - r34, r34_2b, d121 and r101.
    # rosta (Representations of Semantic and Transition areas): we provide three types - TT, FT, and BT.
    # We also present RIM indicates RoSTa Integration Module.
    # bg_choice: original (ORI-Track), hd (COMP-Track, high resolution background, BG20K), coco (COMP-Track,  MS COCO dataset)
    # fg_generate: the way to generate foregrounds and backgrounds in training, closed_form (needs extra fg/bg generation following closed_form method), alpha_blending (no need for extra fg and bg)
    # rssn_denoise: the flag to use extra desnoie images in RSSN in COMP-Track (hd)
    # model_save_dir: path to save the last checkpoint
    # logname: name of the logging files

    # public--------------------
    # TODO 整理带先验的modnet和gfm
    parser.add_argument('--model', type=str, default='bfd',
                        choices=["bfd", "modnet", "gfm", "u2net", "u2netp", "SHM", "FBDM_img"], help="training model")
    # parser.add_argument('--log_path', type=str, default='./checkSave/log.txt',
    #                     help="saving path of logging file")
    parser.add_argument('--loader_mode', type=str, default='bfd',
                        choices=["bfd", "modnet", "gfm", "u2net"], help="training model")
    parser.add_argument('--gpu', type=str, default='2',
                        help="gpu ids")
    parser.add_argument('--data_set', type=str, default='PPM',
                        help="PPM, AM2K, PPT")
    parser.add_argument('--save_file', type=int, default='19',
                        help="saving path of logging file")
    parser.add_argument('--img_size', type=int, default=1024,
                        help="image size for training")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size for training")
    parser.add_argument('--num_worker', type=int, default=8,
                        help="number of threading")
    parser.add_argument('--pretrain_path', type=str, default='',
                        # ./checkSave/bfd/PPM/10/checkSave_bfd/model_best
                        help="loading path for continuing training")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument('--weight', type=float, default=5e-4,
                        help="image size for training")
    parser.add_argument('--port', type=str, default='23547', help="training model")

    parser.add_argument('--in_w_prior', type=bool, default=False,
                        help="is input with prior")
    parser.add_argument('--epoch', type=int, default=1000,
                        help="the number of training epochs")
    parser.add_argument('--aug_mixup', type=bool, default=False,
                        help="augment with mixup, recommend for [bfd, u2net, u2netp]")
    parser.add_argument('--aug_shadow', type=bool, default=True,
                        help="augment with shadow")
    parser.add_argument('--aug_crop', type=bool, default=False,
                        help="augment with shadow")
    parser.add_argument('--val_per_epoch', type=int, default=1,
                        help="validation interval of epoch")
    parser.add_argument('--show_per_epoch', type=int, default=10,
                        help="show interval of epoch")
    parser.add_argument('--save_per_epoch', type=int, default=5,
                        help="save interval of epoch")

    # parser.add_argument('--fg_path', type=str,
    #                     default='/data/wjw/work/matting_set/data/PPM-100/train/image/',
    #                     help="前景图片路径")
    # parser.add_argument('--bg_path', type=str, default='/data/wjw/work/matting_set/data/PPM-100/train/image/',
    #                     help="背景图片路径，u2net不需要")
    # parser.add_argument('--gt_path', type=str,
    #                     default='/data/wjw/work/matting_set/data/PPM-100/train/matte/',
    #                     help="前景对应的标签路径")
    # parser.add_argument('--val_img_path', type=str,
    #                     default='/data/wjw/work/matting_set/data/PPM-100/val/image/',
    #                     help="验证集前景路径")
    # parser.add_argument('--val_gt_path', type=str,
    #                     default='/data/wjw/work/matting_set/data/PPM-100/val/matte/',
    #                     help="验证集标签路径")
    # parser.add_argument('--show_img_path', type=str,
    #                     default='/data/wjw/work/matting_set/data/PPM-100/val/image/',
    #                     help="要预测中间结果的前景图路径")

    # GFM--------------------
    parser.add_argument('--backbone', type=str, required=False, default='r34',
                        choices=["r34", "r34_2b", "d121", "r101"], help="backbone of GFM")
    parser.add_argument('--rosta', type=str, required=False, default='TT', choices=["TT", "FT", "BT", "RIM"],
                        help="representations of semantic and tarnsition areas")
    parser.add_argument('--bgPath_denoise', type=str, required=False, default='')
    parser.add_argument('--fgPath_denoise', type=str, required=False, default='')
    parser.add_argument('--gfm_im_size', type=int, required=False, default=512)

    # bfd--------------------
    parser.add_argument('--stage', type=str, default='refine',
                        choices=["base", "refine"], help="training stage")

    # u2net--------------------
    parser.add_argument('--scale_size', type=int, default=700, help="backbone of GFM")
    parser.add_argument('--crop_size', type=int, default=640, help="representations of semantic and tarnsition areas")

    args = parser.parse_args()

    if args.data_set == 'PPM':
        args.fg_path = '/data/wjw/work/matting_set/data/PPM-100/train/image/'
        args.bg_path = '' # /data/wjw/work/matting_set/BG-20k/train/
        args.gt_path = '/data/wjw/work/matting_set/data/PPM-100/train/matte/'
        args.val_img_path = '/data/wjw/work/matting_set/data/PPM-100/val/image/'
        args.val_gt_path = '/data/wjw/work/matting_set/data/PPM-100/val/matte/'
        args.trimap_path = ''
        args.show_img_path = '/data/wjw/work/matting_set/data/PPM-100/val/image/'
        args.show_gt_path = '/data/wjw/work/matting_set/data/PPM-100/val/matte/'
    elif args.data_set == 'AM2K':
        args.fg_path = '/data/wjw/work/matting_set/data/AM-2K/train/original/'
        args.bg_path = '/data/wjw/work/matting_set/data/AM-2K/train/original/'
        args.gt_path = '/data/wjw/work/matting_set/data/AM-2K/train/mask/'
        args.val_img_path = '/data/wjw/work/matting_set/data/AM-2K/validation/original/'
        args.val_gt_path = '/data/wjw/work/matting_set/data/AM-2K/validation/mask/'
        args.show_img_path = '/data/wjw/work/matting_set/data/AM-2K/validation/original/'
    elif args.data_set == 'PPT':
        args.fg_path = '/data/wjw/work/matting_set/data/PPTs/train/PPTs100_im/'
        args.bg_path = ''
        args.gt_path = '/data/wjw/work/matting_set/data/PPTs/train/PPTs100_gt/'
        args.val_img_path = '/data/wjw/work/matting_set/data/PPTs/val/PPTs100_im/'
        args.val_gt_path = '/data/wjw/work/matting_set/data/PPTs/val/PPTs100_gt/'
        args.trimap_path = ''
        args.show_img_path = '/data/wjw/work/matting_set/data/PPTs/val/PPTs100_im/'

    head_save = './checkSave/' + str(args.model) + '/' + str(args.data_set) + '/' + str(args.save_file)

    # args.log_path = head_save + '/' + 'log.txt'
    args.log_path = head_save + '/log/'

    if args.model == 'modnet':
        args.save_path_img = head_save + '/temp_results_source/'
        args.save_path_model = head_save + '/checkSave_source/'
    elif args.model == 'gfm':
        args.save_path_img = head_save + '/temp_results_gfm/'
        args.save_path_model = head_save + '/checkSave_gfm/'
    elif args.model == 'bfd':
        args.save_path_img = head_save + '/temp_results_bfd/'
        args.save_path_model = head_save + '/checkSave_bfd/'
    elif args.model == 'u2net':
        args.save_path_img = head_save + '/temp_results_u2net/'
        args.save_path_model = head_save + '/checkSave_u2net/'
    else:
        args.save_path_img = head_save + '/temp_results/'
        args.save_path_model = head_save + '/checkSave/'

    if not os.path.exists(args.save_path_img):
        os.makedirs(args.save_path_img)
        # os.mkdir()
    if not os.path.exists(args.save_path_model):
        os.makedirs(args.save_path_model)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    # print(args)
    return args
