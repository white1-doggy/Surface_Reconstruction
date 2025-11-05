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
import argparse
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


def _gather_subjects(source, required_files):
    subjects = []
    missing_inputs = []
    for name in sorted(os.listdir(source)):
        subj_dir = os.path.join(source, name)
        if not os.path.isdir(subj_dir):
            continue
        expected = [os.path.join(subj_dir, f) for f in required_files]
        if all(os.path.exists(path) for path in expected):
            subjects.append(name)
        else:
            missing_inputs.append((name, [f for f, path in zip(required_files, expected) if not os.path.exists(path)]))

    if missing_inputs:
        print('Skipping subjects without required inputs:')
        for name, missing in missing_inputs:
            print(f'  - {name}: missing {", ".join(missing)}')

    return subjects


if __name__ == '__main__':
    from multiprocessing import Pool
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Generate Sobel edge maps for brain volumes')
    parser.add_argument('--source_folder', type=str,
                        default='/public_bme2/bme-wangqian2/wangxy/T1Img',
                        help='Directory containing subject folders with Step01 outputs')
    parser.add_argument('--target_folder', type=str, default=None,
                        help='Directory where edge maps will be written. Defaults to source folder')
    parser.add_argument('--input_filename', type=str, default='brain.nii.gz',
                        help='Filename of the brain intensity image produced by Step01')
    parser.add_argument('--output_filename', type=str, default='brain_sober.nii.gz',
                        help='Filename used when writing the Sobel edge map')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel worker processes to launch')

    args = parser.parse_args()

    source = args.source_folder
    target = args.target_folder if args.target_folder is not None else args.source_folder

    os.makedirs(target, exist_ok=True)

    subjects = _gather_subjects(source, [args.input_filename])

    if len(subjects) == 0:
        print('No subjects with required inputs found. Nothing to process.')
        exit(0)

    pool = Pool(processes=args.num_workers)
    pbar = tqdm(total=len(subjects))
    pbar.set_description('Edge enhancement (Sobel)')
    call_fun = lambda *call_args: update(pbar, *call_args)

    for item in subjects:
        source_img_path = os.path.join(source, item, args.input_filename)
        target_folder = os.path.join(target, item)
        os.makedirs(target_folder, exist_ok=True)
        target_img_path = os.path.join(target_folder, args.output_filename)

        kwargs = {
            'source_img_path': source_img_path,
            'target_img_path': target_img_path,
        }
        pool.apply_async(_SoberEdge, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()


