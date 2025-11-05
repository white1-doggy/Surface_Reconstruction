
import os
import ants
import argparse
import numpy as np

from tqdm import tqdm

file_list_4_volume = ['brain.nii.gz', 'brain.nii.gz', 'tissue_hemi.nii.gz', 'dk-struct.nii.gz', 'wm_lh.nii.gz',
                      'wm_rh.nii.gz', 'aseg.nii.gz']
file_list_4_surf = ['mprage.nii.gz', 'masked.nii.gz', 'tissue_hemi.nii.gz', 'dk-struct.nii.gz', 'lh.nii.gz',
                    'rh.nii.gz', 'aseg.nii.gz']


def _ants_img_info(img_path):
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _padding_and_copy(source, target, item):
    target_folder = os.path.join(target, item)
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    for idx in range(len(file_list_4_surf)):
        source_path = os.path.join(source, item, file_list_4_volume[idx])
        target_path = os.path.join(target, item, file_list_4_surf[idx])
        origin, spacing, direction, img = _ants_img_info(source_path)

        pad_width = ((4, 4), (4, 4), (4, 4))
        img = np.pad(img, pad_width, 'constant')

        # 更新原点：往负方向平移 pad 前缀的体素数 * spacing
        new_origin = [
            origin[0] + pad_width[0][0] * spacing[0],
            origin[1] + pad_width[1][0] * spacing[1],
            origin[2] - pad_width[2][0] * spacing[2]
        ]

        img = ants.from_numpy(img, new_origin, spacing, direction)
        ants.image_write(img, target_path)


def update(pbar, result):
    pbar.update()


def error_back(err):
    print(err)


if __name__ == '__main__':
    import multiprocessing
    from multiprocessing import Pool
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Inference Setting for Hemisphere Tissue Segmentation')
    parser.add_argument('--source_folder', type=str,
                        default='/public_bme/home/liujm/Code/MD_InfanTSurf/TestData/MNBCP000080-43mo',
                        help='target tissue path')
    parser.add_argument('--target_folder', type=str,
                        default='/public_bme/home/liujm/Code/MD_InfanTSurf/TestData/MNBCP000080-43mo-surf',
                        help='target tissue path')

    args = parser.parse_args()

    source = r'C:\Users\Zifeng Lian\Desktop\ABIDE_50002_MRI_MP-RAGE_br_raw_20120830172854796_S164623_I328631/Step01_sMRI_data'
    target = r'C:\Users\Zifeng Lian\Desktop\ABIDE_50002_MRI_MP-RAGE_br_raw_20120830172854796_S164623_I328631/Step02_sMRI_Surf'

    file_list = os.listdir(source)
    file_list.sort()

    pool_num = 1
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
        pool.apply_async(_padding_and_copy, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()