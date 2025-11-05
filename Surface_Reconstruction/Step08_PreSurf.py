import os
import argparse
import SimpleITK as sitk
import numpy as np

from tqdm import tqdm


def get_hemisphere(tissue_hemi, dk_struct):
    lh = np.zeros((dk_struct.shape[0], dk_struct.shape[1], dk_struct.shape[2]), dtype=dk_struct.dtype)
    rh = np.zeros((dk_struct.shape[0], dk_struct.shape[1], dk_struct.shape[2]), dtype=dk_struct.dtype)

    # CSF
    lh[tissue_hemi == 1] = 10
    rh[tissue_hemi == 2] = 10

    # WM & GM
    lh[tissue_hemi == 6] = 250
    lh[tissue_hemi == 5] = 150
    rh[tissue_hemi == 4] = 250
    rh[tissue_hemi == 3] = 150

    # Cerebellum & MidBrain
    cerebellum_mask = dk_struct == 93
    cerebellum_mask += dk_struct == 94
    cerebellum_mask += dk_struct == 95
    cerebellum_mask += dk_struct == 96
    cerebellum_mask += dk_struct == 99
    lh[cerebellum_mask] = 0
    rh[cerebellum_mask] = 0

    # Lateral Ventricle
    Ventricle_L = dk_struct == 55
    Ventricle_L += dk_struct == 57
    Ventricle_L += dk_struct == 53  # Choroid Plexus
    lh[Ventricle_L] = 250
    Ventricle_R = dk_struct == 56
    Ventricle_R += dk_struct == 58
    Ventricle_R += dk_struct == 54  # Choroid Plexus
    rh[Ventricle_R] = 250

    # Subcortical tissues
    subcortical_L = dk_struct == 41
    subcortical_L += dk_struct == 43
    subcortical_L += dk_struct == 45
    subcortical_L += dk_struct == 47
    subcortical_L += dk_struct == 49
    subcortical_L += dk_struct == 51
    lh[subcortical_L] = 250
    subcortical_R = dk_struct == 42
    subcortical_R += dk_struct == 44
    subcortical_R += dk_struct == 46
    subcortical_R += dk_struct == 48
    subcortical_R += dk_struct == 50
    subcortical_R += dk_struct == 52
    rh[subcortical_R] = 250

    # Hippocampus & Amygdala
    Hippo_L = dk_struct == 35
    Hippo_L += dk_struct == 39
    lh[Hippo_L] = 0
    Hippo_R = dk_struct == 36
    Hippo_R += dk_struct == 40
    rh[Hippo_R] = 0

    return lh, rh


def get_aseg(tissue_hemi, dk_struct):
    aseg = np.zeros((dk_struct.shape[0], dk_struct.shape[1], dk_struct.shape[2]), dtype=dk_struct.dtype)

    # WM & GM
    aseg[tissue_hemi == 6] = 2
    aseg[tissue_hemi == 5] = 3
    aseg[tissue_hemi == 4] = 41
    aseg[tissue_hemi == 3] = 42

    # Cerebellum & MidBrain
    aseg[dk_struct == 93] = 7
    aseg[dk_struct == 94] = 46
    aseg[dk_struct == 95] = 8
    aseg[dk_struct == 96] = 47

    midbrain = dk_struct == 99
    aseg[dk_struct == 99] = 174
    aseg[dk_struct == 107] = 173
    aseg[dk_struct == 108] = 175

    sum_axial = np.sum(midbrain, axis=(1, 2))
    total_target_area = np.sum(sum_axial)
    one_third_total = total_target_area / 3
    two_third_total = total_target_area / 3 * 2

    cumulative_sum = np.cumsum(sum_axial)
    axial_one_third = np.searchsorted(cumulative_sum, one_third_total)
    axial_two_third = np.searchsorted(cumulative_sum, two_third_total)
    mask_173, mask_174, mask_175 = midbrain.copy(), midbrain.copy(), midbrain.copy()
    mask_175[axial_one_third:, :, :] = False
    mask_173[:axial_two_third, :, :] = False
    mask_174[:axial_one_third, :, :] = False
    mask_174[axial_two_third:, :, :] = False
    aseg[mask_175] = 175
    aseg[mask_174] = 174
    aseg[mask_173] = 173

    # Lateral Ventricle & Plexus
    aseg[dk_struct == 55] = 4
    aseg[dk_struct == 57] = 5
    aseg[dk_struct == 56] = 43
    aseg[dk_struct == 58] = 44
    aseg[dk_struct == 53] = 31
    aseg[dk_struct == 54] = 63

    # 3rd, 4th Ventricle
    aseg[dk_struct == 97] = 14
    aseg[dk_struct == 98] = 15

    # Subcortical tissues
    aseg[dk_struct == 41] = 11
    aseg[dk_struct == 42] = 50
    aseg[dk_struct == 43] = 12
    aseg[dk_struct == 44] = 51
    aseg[dk_struct == 45] = 13
    aseg[dk_struct == 46] = 52
    aseg[dk_struct == 47] = 9
    aseg[dk_struct == 48] = 48
    aseg[dk_struct == 49] = 26
    aseg[dk_struct == 50] = 58
    aseg[dk_struct == 51] = 28
    aseg[dk_struct == 52] = 60

    # Hippocampus & Amygdala
    aseg[dk_struct == 35] = 17
    aseg[dk_struct == 36] = 53
    aseg[dk_struct == 39] = 18
    aseg[dk_struct == 40] = 54

    # Corpus callosum
    aseg[dk_struct == 102] = 255
    aseg[dk_struct == 103] = 254
    aseg[dk_struct == 104] = 253
    aseg[dk_struct == 105] = 252
    aseg[dk_struct == 106] = 251

    return aseg


def _pre_surf_convert(args, source, target, item):
    source_folder = os.path.join(source, item)
    target_folder = os.path.join(target, item)
    os.makedirs(target_folder, exist_ok=True)

    intensity_path = os.path.join(source_folder, args.intensity_filename)
    tissue_hemi_path = os.path.join(source_folder, args.tissue_hemi_filename)
    dk_path = os.path.join(source_folder, args.dk_filename)
    wm_lh_path = os.path.join(target_folder, args.wm_lh_filename)
    wm_rh_path = os.path.join(target_folder, args.wm_rh_filename)
    aseg_path = os.path.join(target_folder, args.aseg_filename)

    t1w = sitk.ReadImage(intensity_path)
    tissue_hemi = sitk.GetArrayFromImage(sitk.ReadImage(tissue_hemi_path))
    dk_struct = sitk.GetArrayFromImage(sitk.ReadImage(dk_path))

    lh, rh = get_hemisphere(tissue_hemi, dk_struct)
    lh_nii = sitk.GetImageFromArray(lh)
    lh_nii.SetOrigin(t1w.GetOrigin())
    lh_nii.SetSpacing(t1w.GetSpacing())
    lh_nii.SetDirection(t1w.GetDirection())

    rh_nii = sitk.GetImageFromArray(rh)
    rh_nii.SetOrigin(t1w.GetOrigin())
    rh_nii.SetSpacing(t1w.GetSpacing())
    rh_nii.SetDirection(t1w.GetDirection())

    aseg = get_aseg(tissue_hemi, dk_struct)
    aseg_nii = sitk.GetImageFromArray(aseg)
    aseg_nii.SetOrigin(t1w.GetOrigin())
    aseg_nii.SetSpacing(t1w.GetSpacing())
    aseg_nii.SetDirection(t1w.GetDirection())

    sitk.WriteImage(rh_nii, wm_rh_path)
    sitk.WriteImage(lh_nii, wm_lh_path)
    sitk.WriteImage(aseg_nii, aseg_path)


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

    parser = argparse.ArgumentParser(description='Generate hemisphere-specific WM masks and aseg volumes')
    parser.add_argument('--source_folder', type=str,
                        default='/public_bme2/bme-wangqian2/wangxy/T1Img',
                        help='Directory containing subject folders with Step07 outputs')
    parser.add_argument('--target_folder', type=str, default=None,
                        help='Directory where white-matter volumes and aseg will be written. Defaults to source folder')
    parser.add_argument('--intensity_filename', type=str, default='brain.nii.gz',
                        help='Input intensity image filename')
    parser.add_argument('--tissue_hemi_filename', type=str, default='tissue_hemi.nii.gz',
                        help='Input combined tissue+hemisphere label filename')
    parser.add_argument('--dk_filename', type=str, default='dk-struct.nii.gz',
                        help='Input DK structure filename')
    parser.add_argument('--wm_lh_filename', type=str, default='wm_lh.nii.gz',
                        help='Output filename for the left hemisphere WM mask')
    parser.add_argument('--wm_rh_filename', type=str, default='wm_rh.nii.gz',
                        help='Output filename for the right hemisphere WM mask')
    parser.add_argument('--aseg_filename', type=str, default='aseg.nii.gz',
                        help='Output filename for the aseg volume')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel worker processes to launch')

    args = parser.parse_args()

    source = args.source_folder
    target = args.target_folder if args.target_folder is not None else args.source_folder

    os.makedirs(target, exist_ok=True)

    required_files = [args.intensity_filename, args.tissue_hemi_filename, args.dk_filename]
    subjects = _gather_subjects(source, required_files)

    if len(subjects) == 0:
        print('No subjects with required inputs found. Nothing to process.')
        exit(0)

    pool = Pool(processes=args.num_workers)
    pbar = tqdm(total=len(subjects))
    pbar.set_description('Pre-surface volume generation')
    call_fun = lambda *call_args: update(pbar, *call_args)

    for item in subjects:
        kwargs = {
            'args': args,
            'source': source,
            'target': target,
            'item': item,
        }
        pool.apply_async(_pre_surf_convert, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()
