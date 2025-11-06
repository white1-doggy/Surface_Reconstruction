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


def _correction_parallel(args, source, target, item):
    source_folder = os.path.join(source, item)
    target_folder = os.path.join(target, item)
    os.makedirs(target_folder, exist_ok=True)

    tissue_path = os.path.join(source_folder, args.tissue_filename)
    dk_path = os.path.join(source_folder, args.dk_filename)

    hippo_lh_path = os.path.join(target_folder, args.hippo_lh_filename)
    hippo_rh_path = os.path.join(target_folder, args.hippo_rh_filename)

    hippo_lh, hippo_rh = _hippocampus_extraction(dk_path)
    ants.image_write(hippo_lh, hippo_lh_path)
    ants.image_write(hippo_rh, hippo_rh_path)

    if not args.disable_topology_tool:
        os.system('module load compiler/gcc/7.3.1')
        os.system('/public_bme2/bme-dgshen/ZifengLian/BrainSurf/Hippo_Correction/TopologyCorrectionLevelSet --tissue {} --out {}'.format(hippo_lh_path, hippo_lh_path))
        os.system('/public_bme2/bme-dgshen/ZifengLian/BrainSurf/Hippo_Correction/TopologyCorrectionLevelSet --tissue {} --out {}'.format(hippo_rh_path, hippo_rh_path))

    tissue, dk = _hippocampus_correction(tissue_path, dk_path, hippo_lh_path, hippo_rh_path)

    target_tissue_path = os.path.join(target_folder, args.tissue_filename)
    target_dk_path = os.path.join(target_folder, args.dk_filename)

    ants.image_write(tissue, target_tissue_path)
    ants.image_write(dk, target_dk_path)


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
    parser = argparse.ArgumentParser(description='Hippocampus topology correction and DK update')
    parser.add_argument('--source_folder', type=str,
                        default='/public_bme2/bme-wangqian2/wangxy/T1Img',
                        help='Directory containing subject folders with Step02 outputs')
    parser.add_argument('--target_folder', type=str, default=None,
                        help='Directory where corrected outputs will be written. Defaults to source folder')
    parser.add_argument('--tissue_filename', type=str, default='tissue.nii.gz',
                        help='Filename of the tissue segmentation volume')
    parser.add_argument('--dk_filename', type=str, default='dk-struct.nii.gz',
                        help='Filename of the DK atlas segmentation volume')
    parser.add_argument('--hippo_lh_filename', type=str, default='hippo_lh.nii.gz',
                        help='Filename for the temporary left hippocampus volume')
    parser.add_argument('--hippo_rh_filename', type=str, default='hippo_rh.nii.gz',
                        help='Filename for the temporary right hippocampus volume')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel worker processes to launch')
    parser.add_argument('--disable_topology_tool', action='store_true',
                        help='Skip calling the external topology correction binary (for debugging)')

    args = parser.parse_args()

    source = args.source_folder
    target = args.target_folder if args.target_folder is not None else args.source_folder

    os.makedirs(target, exist_ok=True)

    subjects = _gather_subjects(source, [args.tissue_filename, args.dk_filename])

    if len(subjects) == 0:
        print('No subjects with required inputs found. Nothing to process.')
        exit(0)

    pool = Pool(processes=args.num_workers)
    pbar = tqdm(total=len(subjects))
    pbar.set_description('Hippocampus correction')
    call_fun = lambda *call_args: update(pbar, *call_args)

    for item in subjects:
        kwargs = {
            'args': args,
            'source': source,
            'target': target,
            'item': item,
        }
        pool.apply_async(_correction_parallel, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()

