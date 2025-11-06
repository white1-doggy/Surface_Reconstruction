# Handling `.nii.gz` inputs in Step02_pial_surface.sh

`ReplaceWhiteMatter` and several of the downstream topology utilities that are
invoked from `Step02_pial_surface.sh` expect uncompressed NIfTI volumes
(`*.nii`). When the preceding Step09 preparation stage produces gzipped files
such as `lh.nii.gz` and `rh.nii.gz`, those executables abort with ITK
`nifti_image_load failed` or `std::bad_alloc` errors because they are attempting
to read compressed data as if it were uncompressed.

To avoid this failure, make sure each hemisphere volume that is passed into
`Step02_pial_surface.sh` is available as an uncompressed `.nii` file. There are
two straightforward ways to accomplish this:

1. **Decompress before running the script**
   ```bash
   gunzip -c subject/lh.nii.gz > subject/lh.nii
   gunzip -c subject/rh.nii.gz > subject/rh.nii
   gunzip -c subject/lh.topo.nii.gz > subject/lh.topo.nii
   gunzip -c subject/rh.topo.nii.gz > subject/rh.topo.nii
   ```
   After the script finishes you can optionally remove the temporary `.nii`
   copies or recompress them back to `.nii.gz` for storage.

2. **Enable on-the-fly decompression inside the script**
   Modify `Step02_pial_surface.sh` so that, before each call to
   `ReplaceWhiteMatter`, it checks whether the corresponding `.nii` file exists.
   If not, decompress the `.nii.gz` counterpart into a temporary file, call the
   tool with that uncompressed path, and delete or recompress the temporary file
   afterwards. This ensures that reruns stay deterministic even when the source
   data remain gzipped.

Either approach guarantees that `ReplaceWhiteMatter` receives the data format it
expects, preventing the cascading "file doesn't exist" errors later in the
pipeline.
