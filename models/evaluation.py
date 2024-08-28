#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from collections import defaultdict
import numpy as np
from utils import evaluate
import h5py
from tqdm import tqdm
from data import transforms
import torchvision
import nibabel as nib
import os

def evaluate_recon(net_g, data_loader, args, writer = None, epoch = None):
    net_g.eval()
    # testing
    test_logs = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            # input, target, mean, std, norm, fname, slice, max, mask, masked_kspace = batch
            input, target, mean, std, fname, slice, mask, masked_kspace = batch
            mask = mask.to(args.device)
            masked_kspace = masked_kspace.to(args.device)
            output = net_g(input.to(args.device), masked_kspace, mask)

            target = target.to(args.device)

            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            output = transforms.complex_abs(output) * std + mean
            target = transforms.complex_abs(target) * std + mean
            # sum up batch loss
            test_loss = F.l1_loss(output, target)
            test_logs.append({
                'fname': fname,
                'slice': slice,
                'output': (output).cpu().detach().numpy(),
                'target': (target).cpu().numpy(),
                'loss': test_loss.cpu().detach().numpy(),
            })
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        for log in test_logs:
            losses.append(log['loss'])
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])

        for fname in tqdm(outputs):
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['ssim'].append(evaluate.ssim(target, output))
            metrics['psnr'].append(evaluate.psnr(target, output))
        metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        print(metrics, '\n')
        print('No. Volume: ', len(outputs))
        if writer != None:
            writer.add_scalar('Dev_Loss/NMSE', metrics['nmse'], epoch)
            writer.add_scalar('Dev_Loss/SSIM', metrics['ssim'], epoch)
            writer.add_scalar('Dev_Loss/PSNR', metrics['psnr'], epoch)
    return metrics['val_loss'], metrics['nmse'], metrics['ssim'], metrics['psnr']


import matplotlib.pyplot as plt



def test_recon_save(net_g, data_loader, args):
    net_g.eval()
    # testing
    test_logs = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):

            # input, target, mean, std, norm, fname, slice, max, mask, masked_kspace = batch
            input, target, mean, std, fname, slice, mask, masked_kspace = batch
            mask = mask.to(args.device)
            masked_kspace = masked_kspace.to(args.device)

            input = input.to(args.device)
            iter_data_time = time.time()

            output = net_g(input, masked_kspace, mask)
            # output = input.clone()

            t_comp = (time.time() - iter_data_time)
            #print('itr time: ', t_comp)

            target = target.to(args.device)
            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            output = transforms.complex_abs(output) * std + mean
            target = transforms.complex_abs(target) * std + mean
            input = transforms.complex_abs(input) * std + mean

            # sum up batch loss
            test_loss = F.l1_loss(output, target)
            test_logs.append({
                'fname': fname,
                'slice': slice,
                'output': (output).cpu().detach().numpy(),
                'target': (target).cpu().numpy(),
                'input': (input).cpu().numpy(),
                'loss': test_loss.cpu().detach().numpy(),
            })
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        inputs = defaultdict(list)

        figure_dict = {}
        for log in test_logs:
            losses.append(log['loss'])
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                if fname == 'file1000243.h5' and slice.item() == 10:  # fastmri output figure
                    figure_dict['GT'] = log['target'][i]
                    figure_dict['Recon'] = log['output'][i]
                    figure_dict['ZF'] = log['input'][i]
                    figure_dict['note'] = '00250'

                if fname == 'MTR_030.h5' and slice.item() == 50:  # skmtea output figure
                    figure_dict['GT'] = log['target'][i]
                    figure_dict['Recon'] = log['output'][i]
                    figure_dict['ZF'] = log['input'][i]
                    figure_dict['note'] = '00250'

                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
                inputs[fname].append((slice, log['input'][i]))
        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[], lpips=[])
        outputs_save = {}
        targets_save = {}
        input_save = {}
        import lpips
        loss_fn_alex = lpips.LPIPS(net='alex')

        for fname in tqdm(outputs):
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            input = np.stack([tgt for _, tgt in sorted(inputs[fname])])
            outputs_save[fname] = output
            targets_save[fname] = target
            input_save[fname] = input
            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['ssim'].append(evaluate.ssim(target, output))
            metrics['psnr'].append(evaluate.psnr(target, output))
            metrics['lpips'] += evaluate.lpips_case(loss_fn_alex, target, output)
            # metrics['lpips'].append(evaluate.lpips_case(target, output))

        psnr_ave = np.mean(metrics['psnr'])
        ssim_ave = np.mean(metrics['ssim'])
        lpips_ave = np.mean(metrics['lpips'])
        psnr_std = np.std(metrics['psnr'])
        ssim_std = np.std(metrics['ssim'])
        lpips_std = np.std(metrics['lpips'])
        metrics_dict = {}
        metrics_dict['psnr_ave'] = psnr_ave
        metrics_dict['ssim_ave'] = ssim_ave
        metrics_dict['lpips_ave'] = lpips_ave
        metrics_dict['psnr_std'] = psnr_std
        metrics_dict['ssim_std'] = ssim_std
        metrics_dict['lpips_std'] = lpips_std
        print(metrics_dict, '\n')
        print('No. Slices: ', len(outputs))

    return metrics_dict, figure_dict



def test_recon_save_skmtea(net_g, data_loader, args):
    net_g.eval()
    # testing
    test_logs = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):

            if idx != 249:
                continue

            # input, target, mean, std, norm, fname, slice, max, mask, masked_kspace = batch
            input, target, mean, std, fname, slice, mask, masked_kspace = batch
            mask = mask.to(args.device)
            masked_kspace = masked_kspace.to(args.device)

            input = input.to(args.device)
            iter_data_time = time.time()

            output = net_g(input, masked_kspace, mask)
            # output = input.clone()

            t_comp = (time.time() - iter_data_time)
            #print('itr time: ', t_comp)

            target = target.to(args.device)
            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            output = (transforms.complex_abs(output) * std + mean) / 1000
            target = (transforms.complex_abs(target) * std + mean) / 1000
            input = (transforms.complex_abs(input) * std + mean) / 1000

            # sum up batch loss
            test_loss = F.l1_loss(output, target)
            test_logs.append({
                'fname': fname,
                'slice': slice,
                'output': (output).cpu().detach().numpy(),
                'target': (target).cpu().numpy(),
                'input': (input).cpu().numpy(),
                'loss': test_loss.cpu().detach().numpy(),
            })
        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        inputs = defaultdict(list)

        figure_dict = {}
        for log in test_logs:
            losses.append(log['loss'])
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                if fname == 'file1000243.h5' and slice.item() == 10:  # fastmri output figure
                    figure_dict['GT'] = log['target'][i]
                    figure_dict['Recon'] = log['output'][i]
                    figure_dict['ZF'] = log['input'][i]
                    figure_dict['note'] = '00250'

                if fname == 'MTR_030.h5' and slice.item() == 50:  # skmtea output figure
                    figure_dict['GT'] = log['target'][i]
                    figure_dict['Recon'] = log['output'][i]
                    figure_dict['ZF'] = log['input'][i]
                    figure_dict['note'] = '00250'

                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
                inputs[fname].append((slice, log['input'][i]))
        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[], lpips=[])
        outputs_save = {}
        targets_save = {}
        input_save = {}
        import lpips
        loss_fn_alex = lpips.LPIPS(net='alex')

        for fname in tqdm(outputs):
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            input = np.stack([tgt for _, tgt in sorted(inputs[fname])])
            outputs_save[fname] = output
            targets_save[fname] = target
            input_save[fname] = input
            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['ssim'].append(evaluate.ssim(target, output))
            metrics['psnr'].append(evaluate.psnr(target, output))
            metrics['lpips'] += evaluate.lpips_case(loss_fn_alex, target, output)
            # metrics['lpips'].append(evaluate.lpips_case(target, output))

        psnr_ave = np.mean(metrics['psnr'])
        ssim_ave = np.mean(metrics['ssim'])
        lpips_ave = np.mean(metrics['lpips'])
        psnr_std = np.std(metrics['psnr'])
        ssim_std = np.std(metrics['ssim'])
        lpips_std = np.std(metrics['lpips'])
        metrics_dict = {}
        metrics_dict['psnr_ave'] = psnr_ave
        metrics_dict['ssim_ave'] = ssim_ave
        metrics_dict['lpips_ave'] = lpips_ave
        metrics_dict['psnr_std'] = psnr_std
        metrics_dict['ssim_std'] = ssim_std
        metrics_dict['lpips_std'] = lpips_std
        print(metrics_dict, '\n')
        print('No. Slices: ', len(outputs))

    return metrics_dict, figure_dict