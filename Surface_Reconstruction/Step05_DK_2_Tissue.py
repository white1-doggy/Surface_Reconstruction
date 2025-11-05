'''
TODO: Functions to convert SynthSeg dk structure to 3 labels tissue map
@Author: Jiameng Liu
@Create: 09.27.2024, ShanghaiTech University
@Contact: JiamengLiu.PRC@gmail.com
'''
import os
import ants
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython import embed


convert_matrix = pd.read_excel('/public_bme2/bme-dgshen/ZifengLian/BrainSurf/DK_2_Tissue/UII_dk_2_tissue.xlsx')
uAI_dk_index = list(convert_matrix['uAI_Index'])
uAI_tissue_index = list(convert_matrix['Tissue_Index'])


def _ants_img_info(img_path):
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _uii_dk_2_tissue(dk_path, uAI_dk_index, uAI_tissue_index):
    origin, spacing, direction, dk = _ants_img_info(dk_path)

    tissue_tmp = np.zeros_like(dk)
    for idx in range(len(uAI_dk_index)):
        tissue_tmp[dk == uAI_dk_index[idx]] = uAI_tissue_index[idx]

    tissue_tmp = ants.from_numpy(tissue_tmp, origin, spacing, direction)

    return tissue_tmp


def _convert_label(source, target, item):
    target_folder = os.path.join(target, item)
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    source_synthseg_path = os.path.join(source, item, 'dk-struct.nii.gz')
    target_synthseg_path = os.path.join(target, item, 'tissue.nii.gz')
    unified_tissue_synthseg = _uii_dk_2_tissue(source_synthseg_path, uAI_dk_index, uAI_tissue_index)
    ants.image_write(unified_tissue_synthseg, target_synthseg_path)


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
    # file_list = [f for f in file_list if os.path.exists(os.path.join(source, f, 'T2_dk-struct.nii.gz'))]

    pool_num = 8
    pool = Pool(pool_num)

    pbar = tqdm(total=len(file_list))
    pbar.set_description('Tissue convert')
    call_fun = lambda *args: update(pbar, *args)

    for item in file_list:
        kwargs = {
            'source': source,
            'target': target,
            'item': item
        }
        pool.apply_async(_convert_label, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()