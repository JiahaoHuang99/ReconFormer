"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from data import transforms


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(
        #gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=False, data_range=pred.max() - pred.min()
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range = gt.max()
    )


import torch


def float2tensor3(img):

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

def lpips_case(lpips_func, gt, pred):

    _lpips_list = []
    for idx_slice in range(gt.shape[0]):
        gt_slice = gt[idx_slice, :, :]
        pred_slice = pred[idx_slice, :, :]
        v_min = gt_slice.min()
        v_max = gt_slice.max()
        gt_slice = (gt_slice - v_min) / (v_max - v_min)
        gt_slice = float2tensor3(gt_slice)
        gt_slice = (gt_slice * 2 - 1)
        pred_slice = (pred_slice - v_min) / (v_max - v_min)
        pred_slice = float2tensor3(pred_slice)
        pred_slice = (pred_slice * 2 - 1)
        _lpips = lpips_func(gt_slice, pred_slice).data.squeeze().float().cpu().numpy()
        _lpips_list.append(_lpips)

    return _lpips_list


def ssim_slice(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(
        gt, pred, multichannel=False, data_range = gt.max()
    )

def calculate_psnr_single(img1, img2, data_range=None):

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # gt recon
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)


def calculate_ssim_single(img1, img2, data_range=None):

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    return structural_similarity(img1, img2, data_range=data_range)


def calculate_lpips_single(lpips_func, img1, img2):

    gt_slice = float2tensor3(img1)
    gt_slice = (gt_slice * 2 - 1)

    pred_slice = float2tensor3(img2)
    pred_slice = (pred_slice * 2 - 1)
    _lpips = lpips_func(gt_slice, pred_slice).data.squeeze().float().cpu().numpy()

    return _lpips


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )


def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file, 'r') as target, h5py.File(
          args.predictions_path / tgt_file.name, 'r') as recons:
            if args.acquisition and args.acquisition != target.attrs['acquisition']:
                continue

            if args.acceleration and target.attrs['acceleration'] != args.acceleration:
                continue

            target = target[recons_key][()]
            recons = recons['reconstruction'][()]
            target = transforms.center_crop(target, (target.shape[-1], target.shape[-1]))
            recons = transforms.center_crop(recons, (target.shape[-1], target.shape[-1]))
            metrics.push(target, recons)
    return metrics


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--challenge', choices=['singlecoil', 'multicoil'], required=True,
                        help='Which challenge')
    parser.add_argument('--acceleration', type=int, default=None)
    parser.add_argument('--acquisition', choices=['CORPD_FBK', 'CORPDFS_FBK', 'AXT1', 'AXT1PRE',
                        'AXT1POST', 'AXT2', 'AXFLAIR'], default=None,
                        help='If set, only volumes of the specified acquisition type are used '
                             'for evaluation. By default, all volumes are included.')
    args = parser.parse_args()

    recons_key = 'reconstruction_rss' if args.challenge == 'multicoil' else 'reconstruction_esc'
    metrics = evaluate(args, recons_key)
    print(metrics)
