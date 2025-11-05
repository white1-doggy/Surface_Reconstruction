import os
import argparse
import ants
import numpy as np

from tqdm import tqdm
from IPython import embed


def _ants_img_info(img_path):
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _tissue_2_hemi(args, source, target, item):
    source_folder = os.path.join(source, item)
    target_folder = os.path.join(target, item)
    os.makedirs(target_folder, exist_ok=True)

    source_tissue_path = os.path.join(source_folder, args.tissue_filename)
    origin, spacing, direction, tissue = _ants_img_info(source_tissue_path)
    source_hemi_path = os.path.join(source_folder, args.hemi_filename)
    origin, spacing, direction, hemi = _ants_img_info(source_hemi_path)
    target_hemi_path = os.path.join(target_folder, args.output_filename)

    tissue_hemi, tissue_hemi_lh, tissue_hemi_lh_mask, tissue_hemi_rh, tissue_hemi_rh_mask = np.zeros_like(
        tissue), np.zeros_like(tissue), np.zeros_like(tissue), np.zeros_like(tissue), np.zeros_like(tissue)

    tissue_hemi_lh_mask[hemi == 2] = 1
    tissue_hemi_rh_mask[hemi == 1] = 1

    tissue_hemi_lh[tissue == 1] = 1
    tissue_hemi_lh[tissue == 2] = 5
    tissue_hemi_lh[tissue == 3] = 6
    tissue_hemi_lh[tissue_hemi_lh_mask == 0] = 0

    tissue_hemi_rh[tissue == 1] = 2
    tissue_hemi_rh[tissue == 2] = 3
    tissue_hemi_rh[tissue == 3] = 4
    tissue_hemi_rh[tissue_hemi_rh_mask == 0] = 0

    tissue_hemi = tissue_hemi_lh + tissue_hemi_rh
    tissue_hemi[tissue_hemi > 6] = 0

    tissue_hemi = ants.from_numpy(tissue_hemi, origin, spacing, direction)
    ants.image_write(tissue_hemi, target_hemi_path)


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
    import multiprocessing
    from multiprocessing import Pool
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Fuse tissue labels with hemisphere masks')
    parser.add_argument('--source_folder', type=str,
                        default='/public_bme2/bme-wangqian2/wangxy/T1Img',
                        help='Directory containing subject folders with tissue and hemisphere volumes')
    parser.add_argument('--target_folder', type=str, default=None,
                        help='Directory where combined tissue_hemi volumes will be written. Defaults to source folder')
    parser.add_argument('--tissue_filename', type=str, default='tissue.nii.gz',
                        help='Input tissue label filename')
    parser.add_argument('--hemi_filename', type=str, default='hemi.nii.gz',
                        help='Input hemisphere mask filename')
    parser.add_argument('--output_filename', type=str, default='tissue_hemi.nii.gz',
                        help='Filename used when writing the fused output')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel worker processes to launch')

    args = parser.parse_args()

    source = args.source_folder
    target = args.target_folder if args.target_folder is not None else args.source_folder

    os.makedirs(target, exist_ok=True)

    subjects = _gather_subjects(source, [args.tissue_filename, args.hemi_filename])

    if len(subjects) == 0:
        print('No subjects with required inputs found. Nothing to process.')
        exit(0)

    pool = Pool(processes=args.num_workers)
    pbar = tqdm(total=len(subjects))
    pbar.set_description('Tissue + hemisphere fusion')
    call_fun = lambda *call_args: update(pbar, *call_args)

    for item in subjects:
        kwargs = {
            'args': args,
            'source': source,
            'target': target,
            'item': item
        }
        pool.apply_async(_tissue_2_hemi, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()
