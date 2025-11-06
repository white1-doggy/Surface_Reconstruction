import os
import argparse
import numpy as np
import ants

from tqdm import tqdm


def _ants_img_info(img_path):
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _padding_and_copy(args, source, target, item, file_map):
    target_folder = os.path.join(target, item)
    os.makedirs(target_folder, exist_ok=True)

    pad_width = tuple((args.pad_width, args.pad_width) for _ in range(3))

    for input_name, output_name in file_map:
        source_path = os.path.join(source, item, input_name)
        target_path = os.path.join(target_folder, output_name)
        origin, spacing, direction, img = _ants_img_info(source_path)

        padded = np.pad(img, pad_width, mode='constant')

        new_origin = [
            origin[0] + pad_width[0][0] * spacing[0],
            origin[1] + pad_width[1][0] * spacing[1],
            origin[2] - pad_width[2][0] * spacing[2],
        ]

        padded_img = ants.from_numpy(padded, new_origin, spacing, direction)
        ants.image_write(padded_img, target_path)


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

    parser = argparse.ArgumentParser(description='Prepare padded inputs for surface reconstruction')
    parser.add_argument('--source_folder', type=str,
                        default='/public_bme2/bme-wangqian2/wangxy/T1Img',
                        help='Directory containing subject folders with Step08 outputs')
    parser.add_argument('--target_folder', type=str, default=None,
                        help='Directory where padded surface-recon inputs will be written. Defaults to source folder')
    parser.add_argument('--pad_width', type=int, default=4,
                        help='Number of voxels padded on each side for all axes')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel worker processes to launch')
    parser.add_argument('--input_filenames', nargs='+', default=['brain.nii.gz', 'brain.nii.gz', 'tissue_hemi.nii.gz',
                                                                'dk-struct.nii.gz', 'wm_lh.nii.gz', 'wm_rh.nii.gz',
                                                                'aseg.nii.gz'],
                        help='Input filenames expected in each subject folder (order aligned with output_filenames)')
    parser.add_argument('--output_filenames', nargs='+', default=['mprage.nii.gz', 'masked.nii.gz', 'tissue_hemi.nii.gz',
                                                                 'dk-struct.nii.gz', 'lh.nii.gz', 'rh.nii.gz',
                                                                 'aseg.nii.gz'],
                        help='Output filenames written to the target folder (order aligned with input_filenames)')

    args = parser.parse_args()

    if len(args.input_filenames) != len(args.output_filenames):
        raise ValueError('input_filenames and output_filenames must have the same length')

    file_map = list(zip(args.input_filenames, args.output_filenames))

    source = args.source_folder
    target = args.target_folder if args.target_folder is not None else args.source_folder

    os.makedirs(target, exist_ok=True)

    required_files = sorted(set(args.input_filenames))
    subjects = _gather_subjects(source, required_files)

    if len(subjects) == 0:
        print('No subjects with required inputs found. Nothing to process.')
        exit(0)

    pool = Pool(processes=args.num_workers)
    pbar = tqdm(total=len(subjects))
    pbar.set_description('Surface preparation padding')
    call_fun = lambda *call_args: update(pbar, *call_args)

    for item in subjects:
        kwargs = {
            'args': args,
            'source': source,
            'target': target,
            'item': item,
            'file_map': file_map,
        }
        pool.apply_async(_padding_and_copy, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()
