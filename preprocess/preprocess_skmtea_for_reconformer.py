'''
# -----------------------------------------
Preprocess Script for SKM-TEA Dataset (d.0.2)
Dataset: SKM-TEA
Note: No normalisation is applied in this version
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

import os
import h5py
import cv2
from glob import glob
import numpy as np
import json
import imageio
from scipy.fftpack import *
import shutil
import nibabel as nib


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_file_name_from_path(path):
    file_name, ext = os.path.splitext(os.path.basename(path))
    assert ext == '.h5'
    return file_name


def process_h5(data_path,
               save_master_path,
               dataset_type,
               data_type='image_complex',
               is_write=True,):

    # get basic information
    data_name = get_file_name_from_path(data_path)

    if data_type == 'image_complex':

        # read h5 data
        with h5py.File(data_path, 'r') as hf:

            # data_dict = {dataset: hf[dataset][()] for dataset in hf.keys()}
            # attributes = {attr: hf.attrs[attr] for attr in hf.attrs.keys()}

            # print('Keys:', list(hf.keys()))
            data_case = hf['target'][()]  # (512, 512, 160~, 2, 1)
            # masks = hf['masks'][()]  # corresponding poisson sampling mask
            # kspace = hf['kspace'][()]  # raw k-space  (512, 512, 160~, 2, 16)
            # maps = hf['maps'][()] # reconstruction (fully-sampled)  (512, 512, 160~, 16, 1)
            x, y, z, echo_num, _ = data_case.shape
            print(data_case.shape)
            assert echo_num == 2
            assert x == 512
            assert y == 512
            # assert z == 160

        # slice selection
        data_case = data_case[:, :, z//2-50: z//2+50, 0, 0]  # ECHO 0

        h, w, slice_num = data_case.shape

        # loop for slice

        reconstruction_esc_list = []
        kspace_data_list = []
        for slice_idx in range(slice_num):
            img_slice = data_case[..., slice_idx]
            kspace_slice = fftshift(fftn(ifftshift(img_slice, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
            reconstruction_esc_list.append(img_slice)
            kspace_data_list.append(kspace_slice)

        reconstruction_esc = np.stack(reconstruction_esc_list, axis=0)
        kspace_data = np.stack(kspace_data_list, axis=0)

        if is_write:
            mkdir(os.path.join(save_master_path, dataset_type))
            with h5py.File(os.path.join(save_master_path, dataset_type, '{}.h5'.format(data_name)), 'w') as hf_new:
                hf_new.create_dataset('reconstruction_esc', data=reconstruction_esc)
                hf_new.create_dataset('kspace', data=kspace_data)


if __name__ == '__main__':

    data_source_path = '/media/NAS02/SKM-TEA/rawdata/skm-tea/v1-release/files_recon_calib-24'
    mask_source_path = '/media/NAS02/SKM-TEA/rawdata/skm-tea/v1-release/segmentation_masks/raw-data-track'
    ann_file_master_path = "/media/NAS02/SKM-TEA/rawdata/skm-tea/v1-release/annotations/v1.0.0"
    # save_master_path = '/media/NAS03/SKM-TEA/d.0.4.reconformer.mini'
    save_master_path = '/media/NAS03/SKM-TEA/d.0.4.reconformer'
    is_write = True
    # dataset_types = ['train', 'val', 'test']
    dataset_types = ['train', 'test', 'val']

    # Loop for dataset type
    for dataset_type in dataset_types:
        print("#################")
        print("Process dataset type: {}".format(dataset_type))
        ann_file_path = os.path.join(ann_file_master_path, '{}.json'.format(dataset_type))
        ann_new_file_path = os.path.join(save_master_path, '{}.json'.format(dataset_type))
        mkdir(os.path.join(save_master_path))
        shutil.copyfile(ann_file_path, ann_new_file_path)

        with open(ann_new_file_path, "r") as f:
            annotations = json.load(f)
            case_info_list = annotations["images"]
            save_path = os.path.join(save_master_path)

        # Loop for case in dataset
        for case_idx, case_info in enumerate(case_info_list):

            # if case_idx > 5:
            #     break

            file_name_h5 = case_info['file_name']
            file_name_nii = file_name_h5.replace('.h5', '.nii.gz')
            scan_id = case_info['scan_id']
            print('{} / {}: Scan_ID {}'.format(case_idx + 1, len(case_info_list), scan_id))
            case_path = os.path.join(data_source_path, file_name_h5)

            # process for case
            process_h5(data_path=case_path,
                       save_master_path=save_path,
                       dataset_type=dataset_type,
                       data_type='image_complex',
                       is_write=is_write)
