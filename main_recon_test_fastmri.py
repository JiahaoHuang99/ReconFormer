#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import os
from utils.options import args_parser
from models.evaluation import test_recon_save
from data.mri_data import SliceData, DataTransform, DataTransformFixedMask
from data.subsample import create_mask_for_mask_type
from models.Recurrent_Transformer import ReconFormer
import pathlib
from torch.utils.data import DataLoader
from mask.select_mask import define_Mask

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import numpy as np
import skimage.metrics
from utils import evaluate

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    # parse args
    args, parser = args_parser()
    path_dict = {'F': pathlib.Path(args.F_path)}
    resolution_dict = {'F': 320}
    rate_dict = {'F': 1.0}
    args.device = torch.device('cuda:{}'.format(args.gpu[0]) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.resolution = resolution_dict[args.test_dataset]

    # data loader
    def _create_dataset(data_path,data_transform, data_partition, sequence, bs, shuffle, sample_rate=None, display=False):
        sample_rate = sample_rate or args.sample_rate
        dataset = SliceData(
            root=data_path / data_partition,
            transform=data_transform,
            sample_rate=sample_rate,
            challenge=args.challenge,
            sequence=sequence
        )
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle, pin_memory=False, num_workers=8)


    # load dataset and split users
    if args.challenge == 'singlecoil':
        # mask = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
        # val_data_transform = DataTransform(args.resolution, args.challenge, mask, use_seed=True)
        # load mask
        mask_1d = define_Mask('fMRI_Ran_AF{}_CF{}_PE{}'.format(args.accelerations[0], args.center_fractions[0], args.resolution))
        mask = torch.from_numpy(mask_1d[None, :, None]).float()
        val_data_transform = DataTransformFixedMask(args.resolution, args.challenge, mask, use_seed=True)

        if args.phase == 'test':
            dataset_val = _create_dataset(path_dict[args.test_dataset]/args.sequence,val_data_transform, 'val', args.sequence, 1, False, 1.0)
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'ReconFormer':
        net = ReconFormer(in_channels=2, out_channels=2, num_ch=(96, 48, 24),num_iter=5,
        down_scales=(2,1,1.5), img_size=args.resolution, num_heads=(6,6,6), depths=(2,1,1),
        window_sizes=(8,8,8), mlp_ratio=2., resi_connection ='1conv',
        use_checkpoint=(False, False, False, False, False, False)
        ).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net)

    # copy weights
    if len(args.gpu) > 1:
        net = torch.nn.DataParallel(net, args.gpu)

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.phase == 'test':
        if len(args.gpu) > 1:
            # pass
            net.module.load_state_dict(torch.load(args.checkpoint))
        else:
            # pass
            net.load_state_dict(torch.load(args.checkpoint))
        print('Load checkpoint :', args.checkpoint)
        metrics_dict, figure_dict = test_recon_save(net, dataset_val, args)

        import csv

        save_path = os.path.join('results', 'ReconFormer_FastMRI_AF{}'.format(args.accelerations[0]))
        mkdir(save_path)
        with open(os.path.join(save_path, 'results_case_ave.csv'), 'w') as cf:
            writer = csv.writer(cf)
            writer.writerow(['ReconFormer',
                             metrics_dict['ssim_ave'],
                             metrics_dict['ssim_std'],
                             metrics_dict['psnr_ave'],
                             metrics_dict['psnr_std'],
                             metrics_dict['lpips_ave'],
                             metrics_dict['lpips_std'],
                             0])

        import matplotlib.pyplot as plt
        import numpy as np
        import cv2

        img_gt = figure_dict['GT']
        img_gen = figure_dict['Recon']
        img_lq = figure_dict['ZF']
        note = figure_dict['note']
        vmax, vmin = img_gt.max(), img_gt.min()

        img_gt = (img_gt - vmin) / (vmax - vmin)
        img_gen = (img_gen - vmin) / (vmax - vmin)
        img_lq = (img_lq - vmin) / (vmax - vmin)

        # calculate single image metrics
        ssim_slice = evaluate.calculate_ssim_single(img_gt, img_gen, data_range=img_gt.max())
        psnr_slice = evaluate.calculate_psnr_single(img_gt, img_gen, data_range=img_gt.max())
        import lpips
        loss_fn_alex = lpips.LPIPS(net='alex')
        lpips_slice = evaluate.calculate_lpips_single(loss_fn_alex, img_gt, img_gen)

        with open(os.path.join(save_path, 'results_slice_00250.csv'), 'w') as cf:
            writer = csv.writer(cf)
            writer.writerow(['ReconFormer', ssim_slice, psnr_slice, lpips_slice])

        plt.imsave(os.path.join(save_path, 'GT_{}.png'.format(note)), img_gt, cmap='gray')
        plt.imsave(os.path.join(save_path, 'Recon_{}.png'.format(note)), img_gen, cmap='gray')
        plt.imsave(os.path.join(save_path, 'ZF_{}.png'.format(note)), img_lq, cmap='gray')

        diff_gen_x10 = np.clip(np.abs(img_gt - img_gen) * 10, 0, 1)
        diff_lq_x10 = np.clip(np.abs(img_gt - img_lq) * 10, 0, 1)

        diff_gen_x10 = (diff_gen_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8
        diff_lq_x10 = (diff_lq_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8
        diff_gen_x10_color = cv2.applyColorMap(diff_gen_x10, cv2.COLORMAP_JET)
        diff_lq_x10_color = cv2.applyColorMap(diff_lq_x10, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, 'Diff_Recon_{}.png'.format(note)), diff_gen_x10_color)
        cv2.imwrite(os.path.join(save_path, 'Diff_ZF_{}.png'.format(note)), diff_lq_x10_color)
