'''
# -----------------------------------------
Define Undersampling Mask
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

import os
import scipy
import scipy.fftpack
from scipy.io import loadmat
import cv2
import numpy as np


def define_Mask(mask_name):

    if mask_name == 'FULL_Res320':
        mask = np.ones((320, 320))

    # GRAPPA-like (with ACS) Regular Acceleration Factor x Central Fraction x PE (from fastMRI)
    elif mask_name == 'fMRI_Reg_AF2_CF0.16_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af2_cf0.16_pe48.npy'))
    elif mask_name == 'fMRI_Reg_AF2_CF0.16_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af2_cf0.16_pe96.npy'))
    elif mask_name == 'fMRI_Reg_AF2_CF0.16_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af2_cf0.16_pe128.npy'))
    elif mask_name == 'fMRI_Reg_AF2_CF0.16_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af2_cf0.16_pe256.npy'))
    elif mask_name == 'fMRI_Reg_AF2_CF0.16_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af2_cf0.16_pe320.npy'))
    elif mask_name == 'fMRI_Reg_AF2_CF0.16_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af2_cf0.16_pe512.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af4_cf0.08_pe48.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af4_cf0.08_pe96.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af4_cf0.08_pe128.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af4_cf0.08_pe256.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af4_cf0.08_pe320.npy'))
    elif mask_name == 'fMRI_Reg_AF4_CF0.08_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af4_cf0.08_pe512.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af8_cf0.04_pe48.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af8_cf0.04_pe96.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af8_cf0.04_pe128.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af8_cf0.04_pe256.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af8_cf0.04_pe320.npy'))
    elif mask_name == 'fMRI_Reg_AF8_CF0.04_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af8_cf0.04_pe512.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af16_cf0.02_pe48.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af16_cf0.02_pe96.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af16_cf0.02_pe128.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af16_cf0.02_pe256.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af16_cf0.02_pe320.npy'))
    elif mask_name == 'fMRI_Reg_AF16_CF0.02_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'regular', 'regular_af16_cf0.02_pe512.npy'))
        
    # GRAPPA-like (with ACS) Random (Gaussian) Acceleration Factor x Central Fraction x PE (from fastMRI)
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af2_cf0.16_pe48.npy'))
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af2_cf0.16_pe96.npy'))
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af2_cf0.16_pe128.npy'))
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af2_cf0.16_pe256.npy'))
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af2_cf0.16_pe320.npy'))
    elif mask_name == 'fMRI_Ran_AF2_CF0.16_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af2_cf0.16_pe512.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af4_cf0.08_pe48.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af4_cf0.08_pe96.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af4_cf0.08_pe128.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af4_cf0.08_pe256.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af4_cf0.08_pe320.npy'))
    elif mask_name == 'fMRI_Ran_AF4_CF0.08_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af4_cf0.08_pe512.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af8_cf0.04_pe48.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af8_cf0.04_pe96.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af8_cf0.04_pe128.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af8_cf0.04_pe256.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af8_cf0.04_pe320.npy'))
    elif mask_name == 'fMRI_Ran_AF8_CF0.04_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af8_cf0.04_pe512.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE48':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af16_cf0.02_pe48.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE96':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af16_cf0.02_pe96.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE128':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af16_cf0.02_pe128.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE256':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af16_cf0.02_pe256.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE320':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af16_cf0.02_pe320.npy'))
    elif mask_name == 'fMRI_Ran_AF16_CF0.02_PE512':
        mask = np.load(os.path.join('mask', 'fastmri', 'random', 'random_af16_cf0.02_pe512.npy'))

    else:
        raise NotImplementedError('Mask [{:s}] is not defined.'.format(mask_name))

    print('Training model [{:s}] is created.'.format(mask_name))

    return mask
