'''
# -----------------------------------------
Preprocess Script for FastMRI Dataset Multi-Coil & Single-Coil Track
Dataset: FastMRI
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

import os
import h5py
import cv2
from glob import glob
import numpy as np
import math
import imageio
from scipy.fftpack import *



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
               track,
               data_type='kspace',
               is_write=False,
               is_crop_to_320=True):

    # get basic information
    data_name = get_file_name_from_path(data_path)

    if data_type == 'kspace':

        # read data
        with h5py.File(data_path, 'r') as hf:
            # print('Keys:', list(hf.keys()))
            # print('Attrs:', hf.attrs)
            # print(hf.attrs['acquisition'])
            data_dict = {dataset: hf[dataset][()] for dataset in hf.keys()}
            attributes = {attr: hf.attrs[attr] for attr in hf.attrs.keys()}

            kspace_data = hf['kspace'][()]  # (slices, coils, h, w) or (slices, h, w)
            reconstruction_rss = hf['reconstruction_rss'][()]  # (slices, h, w)
            reconstruction_esc = hf['reconstruction_esc'][()]  # (slices, h, w)
            acquisition = hf.attrs['acquisition']

        if track == 'singlecoil':
            slices, h, w = kspace_data.shape
        elif track == 'multicoil':
            slices, coils, h, w = kspace_data.shape
        else:
            raise ValueError

        if acquisition == 'CORPDFS_FBK':
            acq_name = 'PDFS'
        elif acquisition == 'CORPD_FBK':
            acq_name = 'PD'
        else:
            raise ValueError

        # slice selection
        kspace_data = kspace_data[slices // 2 - 10: slices // 2 + 10, ...]
        reconstruction_rss = reconstruction_rss[slices // 2 - 10: slices // 2 + 10, ...]
        reconstruction_esc = reconstruction_esc[slices // 2 - 10: slices // 2 + 10, ...]

        kspace_data_new_list = []

        # update slices
        slices = kspace_data.shape[0]

        # loop for slice
        for slice_idx in range(slices):

            kspace_slice = kspace_data[slice_idx, ...]

            # reminder: when singlecoil track, img_slice here is NOT equal to reconstruction_rss_slice!
            img_slice = fftshift(ifftn(ifftshift(kspace_slice, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

            # crop
            if is_crop_to_320:
                img_slice = img_slice[..., (h // 2 - 160):(h // 2 + 160), (w // 2 - 160):(w // 2 + 160)]
                assert img_slice.shape[-1] == 320 and img_slice.shape[-2] == 320
            else:
                img_slice = img_slice[..., (h // 2 - 320):(h // 2 + 320), (w // 2 - 160):(w // 2 + 160)]
                assert img_slice.shape[-1] == 320 and img_slice.shape[-2] == 640

            assert reconstruction_rss[slice_idx, ...].shape[-1] == 320 and reconstruction_rss[slice_idx, ...].shape[-2] == 320

            kspace_slice_new = fftshift(fftn(ifftshift(img_slice, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
            kspace_data_new_list.append(kspace_slice_new)

        kspace_data_new = np.stack(kspace_data_new_list, axis=0)

        if is_write:
            mkdir(os.path.join(save_master_path, acq_name, dataset_type))
            with h5py.File(os.path.join(save_master_path, acq_name, dataset_type, '{}.h5'.format(data_name)), 'w') as hf_new:
                for dataset_key, dataset_value in data_dict.items():
                    if dataset_key == 'kspace':
                        dataset_value = kspace_data_new
                    if dataset_key == 'reconstruction_rss':
                        dataset_value = reconstruction_rss
                    if dataset_key == 'reconstruction_esc':
                        dataset_value = reconstruction_esc
                    hf_new.create_dataset(dataset_key, data=dataset_value)
                for attr_key, attr_value in attributes.items():
                    hf_new.attrs[attr_key] = attr_value

    elif data_type == 'image_rss':
        raise NotImplementedError
    elif data_type == 'image_esc':
        raise NotImplementedError
    else:
        raise ValueError


if __name__ == '__main__':



    dict_train = {
        "is_write": True,
        "is_mini": False,
        "mini_case_num": None,
        "is_crop_to_320": True,
        "is_sens": False,
        "sens_type": None,  # espirit_sigpy, espirit_official_python
        "track": 'singlecoil',
        "dataset_types": ['train'],
    }

    dict_val = {
        "is_write": True,
        "is_mini": False,
        "mini_case_num": None,
        "is_crop_to_320": True,
        "is_sens": False,
        "sens_type": None,  # espirit_sigpy, espirit_official_python
        "track": 'singlecoil',
        "dataset_types": ['val'],
    }


    configuration_list = []
    configuration_list.append(dict_train)
    configuration_list.append(dict_val)

    # configuration template
    '''
    is_write = True
    is_mini = True
    mini_case_num = 30  # default 30, total 199
    is_crop_to_320 = True
    is_sens = True
    sens_type = 'espirit_sigpy'  # espirit_sigpy, espirit_official_python
    track = 'multicoil'  # 'singlecoil', 'multicoil'
    # dataset_types = ['val', 'train']
    dataset_types = ['val']
    '''

    for config_dict in configuration_list:
        print(config_dict)
        is_write = config_dict['is_write']
        is_mini = config_dict['is_mini']
        mini_case_num = config_dict['mini_case_num']
        is_crop_to_320 = config_dict['is_crop_to_320']
        is_sens = config_dict['is_sens']
        sens_type = config_dict['sens_type']
        track = config_dict['track']
        dataset_types = config_dict['dataset_types']

        # -----------------------------------------
        if track == 'singlecoil':
            assert is_sens is False

        abbr = ""
        if track == 'multicoil':
            abbr = abbr + 'mc'
        elif track == 'singlecoil':
            abbr = abbr + 'sc'
        else:
            raise ValueError
        if is_crop_to_320:
            pass
        else:
            abbr = abbr + '.ori640'

        for dataset_type in dataset_types:

            # save path
            save_master_path = os.path.join('/media/NAS03/fastMRI/knee', f'd.4.0.reconformer.{abbr}')
            mkdir(save_master_path)

            if is_mini:
                dataset_type = f'{dataset_type}_mini'
                N_PDFS = 0
                N_PD = 0

            # load path
            data_master_path = f'/media/NAS05/fastmri_nyu/knee/rawdata/{track}_{dataset_type}'

            data_path_list = glob(os.path.join(data_master_path, '*.h5'))
            data_path_list = sorted(data_path_list)

            for idx_data, data_path in enumerate(data_path_list):

                if is_mini:
                    with h5py.File(data_path) as hf:
                        acquisition = hf.attrs['acquisition']
                    if acquisition == 'CORPDFS_FBK':
                        N_PDFS = N_PDFS + 1
                        if N_PDFS > mini_case_num:
                            print('{} / {}: Passed!: {}'.format(idx_data + 1, len(data_path_list), data_path))
                            continue
                    elif acquisition == 'CORPD_FBK':
                        N_PD = N_PD + 1
                        if N_PD > mini_case_num:
                            print('{} / {}: Passed!: {}'.format(idx_data + 1, len(data_path_list), data_path))
                            continue

                print('{} / {}: Processing: {}'.format(idx_data + 1, len(data_path_list), data_path))

                process_h5(data_path=data_path,
                           save_master_path=save_master_path,
                           dataset_type=dataset_type,
                           track=track,
                           data_type='kspace',
                           is_write=is_write,
                           is_crop_to_320=is_crop_to_320,)
