import warnings
warnings.filterwarnings("ignore")

from torch.optim import lr_scheduler

import logging
from torch import optim
# from option import get_args
from options.Base_option import Base_options
# from datasets.data_util import *
from utils.util import *
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from utils.data_aug import random_crop
from utils.eval import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

# load yaml file by mode
base_option = Base_options()
args = base_option.get_args()
yamls_dict = get_yaml_data('./config/' + args.model + '_config.yaml')
set_yaml_to_args(args, yamls_dict)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_CACHE_PATH'] = '~/.cudacache'


def main_worker(rank, n_pros):
    if not os.path.exists('./checkSave/'):
        os.makedirs('./checkSave/')
    logging.basicConfig(filename=args.log_path + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    writer = SummaryWriter(args.log_path)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:{}'.format(args.port), world_size=n_pros,
                            rank=rank)

    # get the dataset dynamically
    dataset = getPackByNameUtil(py_name='datasets.' + args.loader_mode + '_dataset',
                                object_name=args.loader_mode + '_Dataset')
    # get the trainer dynamically
    train_iter = getPackByNameUtil(py_name='trainers.' + args.model + '_trainer',
                                   object_name=args.model + '_Trainer')
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
    # get the model dynamically
    model = getPackByNameUtil(py_name='models.' + args.model + '_net',
                              object_name=args.model + '_Net')

    label_dataSet = dataset(args, mode='train')
    val_dataset = dataset(args, mode='val')
    show_dataset = dataset(args, mode='show')

    label_train_sampler = torch.utils.data.distributed.DistributedSampler(label_dataSet)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    temp_show_sampler = torch.utils.data.distributed.DistributedSampler(show_dataset)

    label_train_loader = DataLoader(label_dataSet,
                                    shuffle=False,
                                    num_workers=args.num_worker,
                                    batch_size=args.batch_size,
                                    pin_memory=True,
                                    sampler=label_train_sampler
                                    )
    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            num_workers=1,
                            batch_size=1,
                            pin_memory=True,
                            sampler=val_sampler)
    temp_show_loader = DataLoader(show_dataset,
                                  shuffle=False,
                                  num_workers=1,
                                  batch_size=1,
                                  pin_memory=True,
                                  sampler=temp_show_sampler)

    net = model(args).to(rank)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(net,
                                                    device_ids=[rank],
                                                    output_device=rank,
                                                    find_unused_parameters=False)
    # torch.autograd.set_detect_anomaly(True)
    # load pretrained parameters
    if args.pretrain_path != '' and args.pretrain_path is not None:
        print("loading from {}".format(args.pretrain_path))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        saved_state_dict = torch.load(args.pretrain_path, map_location='cpu')#['state_dict']
        new_params = net.state_dict().copy()
        for name, param in new_params.items():
            if (name in saved_state_dict and param.size() == saved_state_dict[name].size()):
                new_params[name].copy_(saved_state_dict[name])
                # print(name)
            elif name[7:] in saved_state_dict and param.size() == saved_state_dict[name[7:]].size():
                new_params[name].copy_(saved_state_dict[name[7:]])
                # print(name[7:])
            elif 'module.' + name in saved_state_dict and param.size() == saved_state_dict['module.' + name].size():
                new_params[name].copy_(saved_state_dict['module.' + name])
                # print('module.' + name)
            else:
                print(name)
        net.load_state_dict(new_params)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))

    """Build cosine learning rate scheduler."""
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                               T_max=(args.epoch - args.warmup_step) * len(label_train_loader),
                                               eta_min=1e-6)

    scaler = GradScaler()
    max_val_loss = 100000
    step = 0
    for epoch in range(args.epoch):
        label_train_sampler.set_epoch(epoch)
        loop = tqdm(enumerate(label_train_loader), total=len(label_train_loader), position=rank)
        loss_epoch = 0
        for (i, label_data) in loop:
            if epoch < args.warmup_step:
                cur_lr = args.lr * step / (args.warmup_step * len(label_train_loader))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cur_lr
            else:
                scheduler.step()
            merge_img = label_data[0]
            merge_alpha = label_data[1]
            merge_trimap = label_data[2]
            merge_fg = label_data[3]
            merge_bg = label_data[4]
            user_map = label_data[5]
            idx = label_data[-1]
            if args.aug_crop:
                merge_img, merge_alpha, merge_trimap, merge_fg, merge_bg = \
                    random_crop(merge_img, merge_alpha, merge_trimap, merge_fg, merge_bg)

            # mixup
            if args.aug_mixup:
                lam = np.random.beta(0.4, 0.4)
                index = torch.randperm(merge_img.shape[0]).cuda()
                merge_img = lam * merge_img + (1 - lam) * merge_img[index]
                merge_alpha = lam * merge_alpha + (1 - lam) * merge_alpha[index]
                merge_trimap = lam * merge_trimap + (1 - lam) * merge_trimap[index]
                merge_fg = lam * merge_fg + (1 - lam) * merge_fg[index]
                merge_bg = lam * merge_bg + (1 - lam) * merge_bg[index]

            net.train()
            optimizer.zero_grad()

            loss_dict = train_iter(net,
                                   merge_img.cuda().float(),
                                   merge_trimap.cuda().float(),
                                   merge_alpha.cuda().float(),
                                   user_map=user_map.cuda().float(),
                                   mode=args.model,
                                   fg=merge_fg.cuda().float(),
                                   bg=merge_bg.cuda().float(),
                                   args=args,
                                   epoch=epoch,
                                   cur_step=step,
                                   total_step=args.epoch * len(label_train_loader))
            assert 'loss' in loss_dict.keys(), 'The keys of loss dict must include [loss].'
            loss = loss_dict['loss']

            if args.amp == True:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            step += 1

            # construct the str of loss items
            log_str_pre = args.model + ' epoch:{}/item:{} '.format(epoch, i)
            loss_items = loss_dict.items()
            loss_epoch += loss.item()
            log_str_loss = 'lr:{:.4f} l_e:{:.4f} '.format(optimizer.state_dict()['param_groups'][0]['lr'],
                                                          loss_epoch / (i + 1))
            # log_str_loss = 'le:{:.4f} '.format(loss_epoch / (i + 1))
            for k, l in loss_items:
                log_str_loss += k + ':{:.3f} '.format(l.item())
            if rank == 0:
                logging.info(log_str_pre + log_str_loss)

                # tensorboard
                writer.add_scalar('training_loss', loss.item(), global_step=step)
            loop.set_description(args.model + '|epoch:{}'.format(epoch))
            loop.set_postfix_str(log_str_loss)
        # scheduler.step()
        # validation
        if (epoch+1) % args.val_per_epoch == 0 and epoch >= 0:
            net.eval()
            with torch.no_grad():
                error_sad_sum = 0
                error_mad_sum = 0
                error_mse_sum = 0
                error_grad_sum = 0
                sad_fg_sum = 0
                sad_bg_sum = 0
                sad_tran_sum = 0
                index = 0
                val_loop = tqdm(enumerate(val_loader), total=len(val_loader), position=rank)
                val_loop.set_description('val|')
                for (i, label_data) in val_loop:
                    label_img = label_data[0].cuda().float()
                    label_alpha = label_data[1].cuda().float()  # .unsqueeze(1)
                    trimap = label_data[2].cuda().float().unsqueeze(1)
                    instance_map = label_data[3].cuda().float()

                    eval_out = evaluater_iter(net,
                                              label_img,
                                              label_alpha,
                                              trimap,
                                              fusion=args.fusion,
                                              interac=args.inter_num)
                    error_sad, error_mad, error_mse, error_grad, sad_fg, sad_bg, sad_tran = eval_out[0], eval_out[1], \
                                                                                            eval_out[2], eval_out[3], \
                                                                                            eval_out[4], eval_out[5], \
                                                                                            eval_out[6]

                    # out = net(label_img)
                    # matte = out[-1]
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

                dist.barrier()
                dist.all_reduce(error_sad_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(error_mad_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(error_mse_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(error_grad_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(sad_fg_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(sad_bg_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(sad_tran_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(index, op=dist.ReduceOp.SUM)

                ave_val_loss = error_mad_sum / index
                ave_error_sad_sum = error_sad_sum / index
                ave_error_mad_sum = error_mad_sum / index
                ave_error_mse_sum = error_mse_sum / index
                ave_error_grad_sum = error_grad_sum / index
                ave_sad_fg_sum = sad_fg_sum / index
                ave_sad_bg_sum = sad_bg_sum / index
                ave_sad_tran_sum = sad_tran_sum / index
                if rank == 0:
                    sum_loss = ave_val_loss
                    if sum_loss < max_val_loss:
                        max_val_loss = sum_loss
                        torch.save(net.state_dict(), args.save_path_model + 'model_best')

                    writer.add_scalar('val_SAD', ave_error_sad_sum, global_step=epoch)
                    writer.add_scalar('val_MAD', ave_error_mad_sum, global_step=epoch)
                    writer.add_scalar('val_MSE', ave_error_mse_sum, global_step=epoch)
                    writer.add_scalar('val_Grad', ave_error_grad_sum, global_step=epoch)
                    writer.add_scalar('val_SAD_F', ave_sad_fg_sum, global_step=epoch)
                    writer.add_scalar('val_SAD_B', ave_sad_bg_sum, global_step=epoch)
                    writer.add_scalar('val_SAD_T', ave_sad_tran_sum, global_step=epoch)

                    logging.info(
                        args.model + "|epoch:{} "
                                     "val best_mad_loss:{:.5f} "
                                     "error_sad:{:.5f} "
                                     "error_mad:{:.5f} "
                                     "error_mse:{:.5f} "
                                     "error_grad:{:.5f} "
                                     "sad_fg:{:.5f} "
                                     "sad_bg:{:.5f} "
                                     "sad_tran:{:.5f}"
                        .format(
                            epoch,
                            max_val_loss,
                            ave_error_sad_sum,
                            ave_error_mad_sum,
                            ave_error_mse_sum,
                            ave_error_grad_sum,
                            ave_sad_fg_sum,
                            ave_sad_bg_sum,
                            ave_sad_tran_sum)
                    )

                    metrix_str = '{:20}\t{:20}\t{:20}\n' \
                                 '{:20}\t{:20}\t{:20}\n' \
                                 '{:20}\t{:20}\t{:20}\n' \
                        .format('Val|epoch: {}'.format(epoch),
                                'Best_mad: {:.5f}'.format(max_val_loss),
                                'Grad: {:.5f}'.format(ave_error_grad_sum),
                                'Sad: {:.5f}'.format(ave_error_sad_sum),
                                'Mad: {:.5f}'.format(ave_error_mad_sum),
                                'Mse: {:.5f}'.format(ave_error_mse_sum),
                                'Sad_fg: {:.5f}'.format(ave_sad_fg_sum),
                                'Sad_bg: {:.5f}'.format(ave_sad_bg_sum),
                                'Sad_tran: {:.5f}'.format(ave_sad_tran_sum)
                                )
                    print(metrix_str)

        if (epoch+1) % args.show_per_epoch == 0 and epoch > 0:
            net.eval()
            with torch.no_grad():
                for (i, label_data) in enumerate(temp_show_loader):
                    if i == 100:
                        break
                    label_img = label_data[0].cuda().float()
                    label_alpha = label_data[1].cuda().float()

                    matte, source = shower_iter(net, label_img, label_alpha)
                    source = (source.cpu().numpy()[0].transpose([1, 2, 0])) * 255

                    # out = net(label_img.cuda().float())
                    # matte = out[-1]
                    # source = (label_img.cuda().float().cpu().numpy()[0].transpose([1, 2, 0])) * 255

                    source = np.clip(source, 0, 255)[..., :3]

                    matte = matte[0][0].data.cpu().numpy()
                    Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(
                        args.save_path_img + '{}_mask.png'.format(i))

                    bg = np.zeros(shape=source.shape)
                    bg[:, :, 1] = 255
                    merge = source * matte[:, :, np.newaxis] + bg * (1 - matte[:, :, np.newaxis])
                    merge = np.array(merge, dtype='uint8')
                    cv2.imwrite(args.save_path_img + '{}.png'.format(i), merge)

        if epoch % args.save_per_epoch == 0 and rank == 0 and epoch > 0:
            torch.save(net.state_dict(), args.save_path_model + 'model_{}'.format(epoch))


if __name__ == '__main__':
    DDP = True
    mp.spawn(main_worker, nprocs=len(args.gpu.split(',')), args=(len(args.gpu.split(',')),))
