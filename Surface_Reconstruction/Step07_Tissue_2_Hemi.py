import os
import ants
import numpy as np

from tqdm import tqdm
from IPython import embed


def _ants_img_info(img_path):
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _tissue_2_hemi(source, target, item):
    source_tissue_path = os.path.join(source, item, 'tissue.nii.gz')
    origin, spacing, direction, tissue = _ants_img_info(source_tissue_path)
    source_hemi_path = os.path.join(source, item, 'hemi.nii.gz')
    origin, spacing, direction, hemi = _ants_img_info(source_hemi_path)
    target_hemi_path = os.path.join(target, item, 'tissue_hemi.nii.gz')

    tissue_hemi, tissue_hemi_lh, tissue_hemi_lh_mask, tissue_hemi_rh, tissue_hemi_rh_mask = np.zeros_like(
        tissue), np.zeros_like(tissue), np.zeros_like(tissue), np.zeros_like(tissue), np.zeros_like(tissue)

    tissue_hemi_lh_mask[hemi==2] = 1
    tissue_hemi_rh_mask[hemi==1] = 1

    tissue_hemi_lh[tissue==1] = 1
    tissue_hemi_lh[tissue==2] = 5
    tissue_hemi_lh[tissue==3] = 6
    tissue_hemi_lh[tissue_hemi_lh_mask==0] = 0

    tissue_hemi_rh[tissue==1] = 2
    tissue_hemi_rh[tissue==2] = 3
    tissue_hemi_rh[tissue==3] = 4
    tissue_hemi_rh[tissue_hemi_rh_mask==0] = 0

    tissue_hemi = tissue_hemi_lh+tissue_hemi_rh
    tissue_hemi[tissue_hemi>6] = 0

    tissue_hemi = ants.from_numpy(tissue_hemi, origin, spacing, direction)
    ants.image_write(tissue_hemi, target_hemi_path)


def update(pbar, result):
    pbar.update()


def error_back(err):
    print(err)


if __name__ == '__main__':
    import multiprocessing
    from multiprocessing import Pool
    from tqdm import tqdm

    source = '/home_data/home/lianzf2024/test'
    target = '/home_data/home/lianzf2024/test'

    file_list = os.listdir(source)
    file_list.sort()

    pool_num = 8
    pool = Pool(pool_num)
    pbar = tqdm(total=len(file_list))
    pbar.set_description('Hemi Sphere Tissue Update')
    call_fun = lambda *args: update(pbar, *args)

    for item in file_list:
        kwargs = {
            'source': source,
            'target': target,
            'item': item
        }
        pool.apply_async(_tissue_2_hemi, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()