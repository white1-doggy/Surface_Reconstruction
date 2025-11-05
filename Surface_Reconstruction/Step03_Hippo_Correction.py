import os
import ants
import argparse
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool


def _ants_img_info(img_path):
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _hippocampus_extraction(dk_path):
    origin, spacing, direction, dk = _ants_img_info(dk_path)
    hippo_lh, hippo_rh = np.zeros_like(dk), np.zeros_like(dk)
    hippo_lh[dk==35] = 250
    hippo_rh[dk==36] = 250

    hippo_lh = ants.from_numpy(hippo_lh, origin, spacing, direction)
    hippo_rh = ants.from_numpy(hippo_rh, origin, spacing, direction)

    return hippo_lh, hippo_rh


def _hippocampus_correction(tissue_path, dk_path, hippo_lh, hippo_rh):
    origin, spacing, direction, tissue = _ants_img_info(tissue_path)
    origin, spacing, direction, dk = _ants_img_info(dk_path)
    origin, spacing, direction, hippo_lh = _ants_img_info(hippo_lh)
    origin, spacing, direction, hippo_rh = _ants_img_info(hippo_rh)

    tissue[hippo_lh!=0] = 2
    tissue[hippo_rh!=0] = 2
    tissue = ants.from_numpy(tissue, origin, spacing, direction)

    dk[hippo_lh!=0] = 35
    dk[hippo_rh!=0] = 36
    dk = ants.from_numpy(dk, origin, spacing, direction)

    return tissue, dk


def _correction_parallel(tissue_path, dk_path, hippo_lh_path, hippo_rh_path):
    hippo_lh, hippo_rh = _hippocampus_extraction(dk_path)
    ants.image_write(hippo_lh, hippo_lh_path)
    ants.image_write(hippo_rh, hippo_rh_path)

    os.system('module load compiler/gcc/7.3.1')
    os.system('/public_bme2/bme-dgshen/ZifengLian/BrainSurf/Hippo_Correction/TopologyCorrectionLevelSet --tissue {} --out {}'.format(hippo_lh_path, hippo_lh_path))
    os.system('/public_bme2/bme-dgshen/ZifengLian/BrainSurf/Hippo_Correction/TopologyCorrectionLevelSet --tissue {} --out {}'.format(hippo_rh_path, hippo_rh_path))

    tissue, dk = _hippocampus_correction(tissue_path, dk_path, hippo_lh_path, hippo_rh_path)
    ants.image_write(tissue, tissue_path)
    ants.image_write(dk, dk_path)


def update(pbar, result):
    pbar.update()


def error_back(err):
    print(err)


if __name__ == '__main__':
    from IPython import embed

    source = '/home_data/home/lianzf2024/test'

    file_list = os.listdir(source)
    file_list.sort()

    pool_num = 8
    pool = Pool(pool_num)
    pbar = tqdm(total=len(file_list))
    pbar.set_description('Hippocampus Correction')
    call_fun = lambda *args: update(pbar, *args)

    for item in file_list:
        kwargs = {
            'tissue_path':os.path.join(source, item, 'tissue.nii.gz'),
            'dk_path':os.path.join(source, item, 'dk-struct.nii.gz'),
            'hippo_lh_path':os.path.join(source, item, 'hippo_lh.nii.gz'),
            'hippo_rh_path':os.path.join(source, item, 'hippo_rh.nii.gz')
            }
        pool.apply_async(_correction_parallel, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()

