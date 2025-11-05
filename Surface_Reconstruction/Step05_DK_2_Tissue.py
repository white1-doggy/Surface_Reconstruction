import os
import argparse
import ants
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython import embed


def _load_mapping(mapping_excel):
    convert_matrix = pd.read_excel(mapping_excel)
    return list(convert_matrix['uAI_Index']), list(convert_matrix['Tissue_Index'])


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


def _convert_label(source, target, item, dk_filename, tissue_filename, mapping):
    target_folder = os.path.join(target, item)
    os.makedirs(target_folder, exist_ok=True)
    source_synthseg_path = os.path.join(source, item, dk_filename)
    target_synthseg_path = os.path.join(target_folder, tissue_filename)
    uAI_dk_index, uAI_tissue_index = mapping
    unified_tissue_synthseg = _uii_dk_2_tissue(source_synthseg_path, uAI_dk_index, uAI_tissue_index)
    ants.image_write(unified_tissue_synthseg, target_synthseg_path)


def update(pbar, result):
    pbar.update()


def error_back(err):
    print(err)


def _gather_subjects(source, dk_filename):
    subjects = []
    missing_inputs = []
    for name in sorted(os.listdir(source)):
        subj_dir = os.path.join(source, name)
        if not os.path.isdir(subj_dir):
            continue
        expected = os.path.join(subj_dir, dk_filename)
        if os.path.exists(expected):
            subjects.append(name)
        else:
            missing_inputs.append(name)

    if missing_inputs:
        print('Skipping subjects without required inputs:')
        for name in missing_inputs:
            print(f'  - {name}: missing {dk_filename}')

    return subjects


if __name__ == '__main__':
    import multiprocessing
    from multiprocessing import Pool
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Convert DK structures to unified tissue labels')
    parser.add_argument('--source_folder', type=str,
                        default='/public_bme2/bme-wangqian2/wangxy/T1Img',
                        help='Directory containing subject folders with DK segmentations')
    parser.add_argument('--target_folder', type=str, default=None,
                        help='Directory where tissue volumes will be written. Defaults to source folder')
    parser.add_argument('--dk_filename', type=str, default='dk-struct.nii.gz',
                        help='Filename of the DK segmentation volume to convert')
    parser.add_argument('--tissue_filename', type=str, default='tissue.nii.gz',
                        help='Filename used when writing the converted tissue volume')
    parser.add_argument('--mapping_excel', type=str,
                        default='/public_bme2/bme-dgshen/ZifengLian/BrainSurf/DK_2_Tissue/UII_dk_2_tissue.xlsx',
                        help='Excel file describing the DK-to-tissue mapping')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel worker processes to launch')

    args = parser.parse_args()

    source = args.source_folder
    target = args.target_folder if args.target_folder is not None else args.source_folder

    os.makedirs(target, exist_ok=True)

    subjects = _gather_subjects(source, args.dk_filename)

    if len(subjects) == 0:
        print('No subjects with required inputs found. Nothing to process.')
        exit(0)

    mapping = _load_mapping(args.mapping_excel)

    pool = Pool(processes=args.num_workers)

    pbar = tqdm(total=len(subjects))
    pbar.set_description('DK to tissue conversion')
    call_fun = lambda *call_args: update(pbar, *call_args)

    for item in subjects:
        kwargs = {
            'source': source,
            'target': target,
            'item': item,
            'dk_filename': args.dk_filename,
            'tissue_filename': args.tissue_filename,
            'mapping': mapping,
        }
        pool.apply_async(_convert_label, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()
