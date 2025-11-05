"""
TODO: Code for converting T1w image to edge maps of brain tissue
#    Copyright IDEA Lab, School of Biomedical Engineering, ShanghaiTech University, Shanghai, China
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0

@Create at: 20241030, ShanghaiTech University
@Author: Jiameng Liu
@Contact: JiamengLiu.PRC@gmail.com
"""
import os
import ants
import numpy as np
import SimpleITK as sitk
from IPython import embed
from scipy import ndimage as ndi
from skimage import feature
import SimpleITK as sitk
from tqdm import tqdm

def _ants_img_info(img_path):
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _normalize_z_score(data, clip=True):
    '''
    funtions to normalize data to standard distribution using (data - data.mean()) / data.std()
    :param data: numpy array
    :param clip: whether using upper and lower clip
    :return: normalized data by using z-score
    '''
    if clip == True:
        bounds = np.percentile(data, q=[0.00, 99.999])
        data[data <= bounds[0]] = bounds[0]
        data[data >= bounds[1]] = bounds[1]

    return (data - data.mean()) / data.std()


def _SoberEdge(source_img_path, target_img_path):
    # TODO: normalize data according to z-score strategy
    origin, spacing, direction, img = _ants_img_info(source_img_path)
    img = _normalize_z_score(img)
    img = ants.from_numpy(img, origin, spacing, direction)
    ants.image_write(img, target_img_path)

    # TODO: Generate edge map through Sober Operator
    data_nii = sitk.ReadImage(target_img_path)
    origin = data_nii.GetOrigin()
    spacing = data_nii.GetSpacing()
    direction = data_nii.GetDirection()

    data_float_nii = sitk.Cast(data_nii, sitk.sitkFloat32)

    sobel_op = sitk.SobelEdgeDetectionImageFilter()
    sobel_sitk = sobel_op.Execute(data_float_nii)
    sobel_sitk = sitk.Cast(sobel_sitk, sitk.sitkInt16)

    sobel_sitk.SetOrigin(origin)
    sobel_sitk.SetSpacing(spacing)
    sobel_sitk.SetDirection(direction)

    sitk.WriteImage(sobel_sitk, target_img_path)
    return None


def update(pbar, result):
    pbar.update()


def error_back(err):
    print(err)


if __name__ == '__main__':
    from multiprocessing import Pool
    from tqdm import tqdm

    source = r'/home_data/home/lianzf2024/test'
    target = r'/home_data/home/lianzf2024/test'

    file_list = os.listdir(source)
    file_list.sort()

    pool_num = 8
    pool = Pool(pool_num)
    pbar = tqdm(total=len(file_list))
    pbar.set_description('Persudo Brain Extraction')
    call_fun = lambda *args: update(pbar, *args)

    for item in file_list:
        source_img_path = os.path.join(source, str(item), 'brain.nii.gz')
        target_img_path = os.path.join(target, str(item), 'brain_sober.nii.gz')

        kwargs = {
            'source_img_path': source_img_path,
            'target_img_path': target_img_path,
        }
        pool.apply_async(_SoberEdge, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()


