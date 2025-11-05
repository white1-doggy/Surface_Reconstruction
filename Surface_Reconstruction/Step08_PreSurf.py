'''2024-05-21
Input: [tissue_hemi, dk_struct]
Output: [lh, rh]
'''

import os
import argparse
import SimpleITK as sitk
import numpy as np
import os, shutil
from tqdm import tqdm


def get_hemisphere(tissue_hemi, dk_struct):
    lh = np.zeros((dk_struct.shape[0], dk_struct.shape[1], dk_struct.shape[2]), dtype=dk_struct.dtype)
    rh = np.zeros((dk_struct.shape[0], dk_struct.shape[1], dk_struct.shape[2]), dtype=dk_struct.dtype)

    # CSF
    lh[tissue_hemi==1] = 10
    rh[tissue_hemi==2] = 10
    # rh[lh==10] = 10
    # lh[:,:, 122:] = 0
    # rh[:,:, :118] = 0

    # WM & GM
    lh[tissue_hemi==6] = 250
    lh[tissue_hemi==5] = 150
    rh[tissue_hemi==4] = 250
    rh[tissue_hemi==3] = 150

    # Cerebellum & MidBrain
    cerebellum_mask = dk_struct==93
    cerebellum_mask += dk_struct==94
    cerebellum_mask += dk_struct==95
    cerebellum_mask += dk_struct==96
    cerebellum_mask += dk_struct==99
    lh[cerebellum_mask] = 0
    rh[cerebellum_mask] = 0

    # Lateral Ventricle
    Ventricle_L = dk_struct== 55
    Ventricle_L += dk_struct== 57
    # Choroid_Plexus
    Ventricle_L += dk_struct== 53
    lh[Ventricle_L] = 250
    Ventricle_R = dk_struct== 56
    Ventricle_R += dk_struct== 58
    # Choroid_Plexus
    Ventricle_R += dk_struct== 54
    rh[Ventricle_R] = 250

    # Subcortical tissues
    subcortical_L = dk_struct== 41
    subcortical_L += dk_struct== 43
    subcortical_L += dk_struct== 45
    subcortical_L += dk_struct== 47
    subcortical_L += dk_struct== 49
    subcortical_L += dk_struct== 51
    lh[subcortical_L] = 250
    subcortical_R = dk_struct== 42
    subcortical_R += dk_struct== 44
    subcortical_R += dk_struct== 46
    subcortical_R += dk_struct== 48
    subcortical_R += dk_struct== 50
    subcortical_R += dk_struct== 52
    rh[subcortical_R] = 250

    # Hippocampus & Amygdala
    Hippo_L = dk_struct== 35
    Hippo_L += dk_struct== 39
    lh[Hippo_L] = 0
    Hippo_R = dk_struct== 36
    Hippo_R += dk_struct== 40
    rh[Hippo_R] = 0

    return lh, rh


def get_aseg(tissue_hemi, dk_struct):
    aseg = np.zeros((dk_struct.shape[0], dk_struct.shape[1], dk_struct.shape[2]), dtype=dk_struct.dtype)

    # WM & GM
    aseg[tissue_hemi==6] = 2
    aseg[tissue_hemi==5] = 3
    aseg[tissue_hemi==4] = 41
    aseg[tissue_hemi==3] = 42

    # Cerebellum & MidBrain
    aseg[dk_struct==93] = 7
    aseg[dk_struct==94] = 46
    aseg[dk_struct==95] = 8
    aseg[dk_struct==96] = 47

    midbrain = dk_struct==99
    aseg[dk_struct==99] = 174
    aseg[dk_struct==107] = 173
    aseg[dk_struct==108] = 175

    sum_axial = np.sum(midbrain, axis=(1,2))
    total_target_area = np.sum(sum_axial)
    one_third_total = total_target_area/3
    two_third_total = total_target_area/3 *2

    cumulative_sum = np.cumsum(sum_axial)
    axial_one_third = np.searchsorted(cumulative_sum, one_third_total)
    axial_two_third = np.searchsorted(cumulative_sum, two_third_total)
    mask_173, mask_174, mask_175 = midbrain.copy(), midbrain.copy(), midbrain.copy()
    mask_175[axial_one_third:, :, :] = False
    mask_173[:axial_two_third, :, :] = False
    mask_174[:axial_one_third, :, :] = False
    mask_174[axial_two_third:, :, :] = False
    aseg[mask_175] = 175
    aseg[mask_174] = 174
    aseg[mask_173] = 173

    # Lateral Ventricle
    aseg[dk_struct==55]=4
    aseg[dk_struct==57]=5
    aseg[dk_struct==56]=43
    aseg[dk_struct==58]=44
    # Choroid_Plexus
    aseg[dk_struct==53]=31
    aseg[dk_struct==54]=63
    # 3rd, 4th Ventricle
    aseg[dk_struct==97]=14
    aseg[dk_struct==98]=15

    # Subcortical tissues
    # Caudate
    aseg[dk_struct==41]=11
    aseg[dk_struct==42]=50
    # Putamen
    aseg[dk_struct==43]=12
    aseg[dk_struct==44]=51
    # Pallidum
    aseg[dk_struct==45]=13
    aseg[dk_struct==46]=52
    # Thalamus
    aseg[dk_struct==47]=9
    aseg[dk_struct==48]=48
    # Accumbens
    aseg[dk_struct==49]=26
    aseg[dk_struct==50]=58
    # Ventricle DC
    aseg[dk_struct==51]=28
    aseg[dk_struct==52]=60

    # Hippocampus & Amygdala
    aseg[dk_struct==35]=17
    aseg[dk_struct==36]=53
    aseg[dk_struct==39]=18
    aseg[dk_struct==40]=54

    # 胼胝体
    aseg[dk_struct==102]=255
    aseg[dk_struct==103]=254
    aseg[dk_struct==104]=253
    aseg[dk_struct==105]=252
    aseg[dk_struct==106]=251

    return aseg


def _PreSurf_Convert(source, item):
    intensity_path = os.path.join(source, item, 'brain.nii.gz')
    tissue_hemi_path = os.path.join(source, item, 'tissue_hemi.nii.gz')
    dk_path = os.path.join(source, item, 'dk-struct.nii.gz')
    wm_lh_path = os.path.join(source, item, 'wm_lh.nii.gz')
    wm_rh_path = os.path.join(source, item, 'wm_rh.nii.gz')
    aseg_path = os.path.join(source, item, 'aseg.nii.gz')

    T1W_nii = sitk.ReadImage(intensity_path)
    tissue_hemi = sitk.GetArrayFromImage(sitk.ReadImage(tissue_hemi_path))
    dk_struct = sitk.GetArrayFromImage(sitk.ReadImage(dk_path))

    lh, rh = get_hemisphere(tissue_hemi, dk_struct)
    lh_nii = sitk.GetImageFromArray(lh)
    lh_nii.SetOrigin(T1W_nii.GetOrigin())
    lh_nii.SetSpacing(T1W_nii.GetSpacing())
    lh_nii.SetDirection(T1W_nii.GetDirection())

    rh_nii = sitk.GetImageFromArray(rh)
    rh_nii.SetOrigin(T1W_nii.GetOrigin())
    rh_nii.SetSpacing(T1W_nii.GetSpacing())
    rh_nii.SetDirection(T1W_nii.GetDirection())

    aseg = get_aseg(tissue_hemi, dk_struct)
    aseg_nii = sitk.GetImageFromArray(aseg)
    aseg_nii.SetOrigin(T1W_nii.GetOrigin())
    aseg_nii.SetSpacing(T1W_nii.GetSpacing())
    aseg_nii.SetDirection(T1W_nii.GetDirection())

    sitk.WriteImage(rh_nii, wm_rh_path)
    sitk.WriteImage(lh_nii, wm_lh_path)
    sitk.WriteImage(aseg_nii, aseg_path)


def update(pbar, result):
    pbar.update()


def error_back(err):
    print(err)


if __name__ == '__main__':
    import multiprocessing
    from multiprocessing import Pool
    from tqdm import tqdm

    source = '/home_data/home/lianzf2024/test'
    file_list = os.listdir(source)
    file_list.sort()

    pool_num = 8
    pool = Pool(pool_num)
    pbar = tqdm(total=len(file_list))
    pbar.set_description('AutoBET: Automatic Brain Extraction Tool')
    call_fun = lambda *args: update(pbar, *args)

    for item in file_list:
        kwargs = {
            'source':source,
            'item':item
        }
        pool.apply_async(_PreSurf_Convert, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()