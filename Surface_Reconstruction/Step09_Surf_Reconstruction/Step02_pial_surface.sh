#!/bin/bash
#SBATCH --job-name=pial_surface
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --partition=bme_cpu
#SBATCH -t 240:00:00

# Fail fast so we can surface the first command that does not produce
# its expected outputs (e.g. *.correct.sulci.nii.gz).
set -euo pipefail

module load compiler/gcc/7.3.1

# Allow overriding the toolchain and subject root via environment variables so
# the script can run in different installations without editing the file.
SCRIPTS_DIR=${SCRIPTS_DIR:-'/public_bme/home/liujm/Code/MD_InfanTSurf/Scripts/Step8_Surf_Recons/'}
FILE_ROOT=${FILE_ROOT:-"/home_data/home/lianzf2024/test2/Step02_sMRI_Surf"}

if [ ! -d "${SCRIPTS_DIR}" ]; then
  echo "[ERROR] SCRIPTS_DIR=${SCRIPTS_DIR} does not exist or is not a directory." >&2
  exit 1
fi

if [ ! -d "${FILE_ROOT}" ]; then
  echo "[ERROR] FILE_ROOT=${FILE_ROOT} does not exist or is not a directory." >&2
  exit 1
fi

cd "${FILE_ROOT}"
echo "[INFO] Processing subjects under ${FILE_ROOT}" >&2

TEMP_FILES=()
cleanup_temp_files() {
  for path in "${TEMP_FILES[@]}"; do
    if [ -n "${path}" ] && [ -f "${path}" ]; then
      rm -f "${path}"
    fi
  done
}
trap cleanup_temp_files EXIT

check_exists() {
  if [ ! -e "$1" ]; then
    echo "[ERROR] Expected file $1 is missing." >&2
    exit 1
  fi
}

prepare_nifti_input() {
  local base="$1"    # path without extension (e.g. subj/lh)
  local path_var="$2" # name of variable to store resolved path

  if [ -f "${base}.nii" ]; then
    eval "${path_var}='${base}.nii'"
  elif [ -f "${base}.nii.gz" ]; then
    local prefix tmp_dir template tmp_path
    prefix="$(basename "${base}")"
    tmp_dir="${TMPDIR:-/tmp}"
    template="${tmp_dir}/${prefix}.XXXXXX.nii"
    if ! tmp_path="$(mktemp "${template}")"; then
      echo "[ERROR] Failed to create temporary file for ${base}.nii.gz" >&2
      exit 1
    fi
    echo "[INFO] Decompressing ${base}.nii.gz -> ${tmp_path}" >&2
    if ! gunzip -c "${base}.nii.gz" > "${tmp_path}"; then
      echo "[ERROR] Failed to decompress ${base}.nii.gz" >&2
      rm -f "${tmp_path}"
      exit 1
    fi
    TEMP_FILES+=("${tmp_path}")
    eval "${path_var}='${tmp_path}'"
  else
    echo "[ERROR] Neither ${base}.nii nor ${base}.nii.gz exists." >&2
    exit 1
  fi
}

gzip_if_exists() {
  for file in "$@"; do
    if [ -f "${file}" ]; then
      if ! gzip -f "${file}"; then
        echo "[ERROR] Failed to gzip ${file}" >&2
        exit 1
      fi
    fi
  done
}

for subject_dir in */; do
  subject=${subject_dir%/}
  echo "[INFO] Subject ${subject}: starting pial reconstruction" >&2
  for hemi in lh rh; do
    check_exists "$subject/$hemi.orig.vtk"

    tissue_path=""
    topo_path=""
    prepare_nifti_input "$subject/$hemi" tissue_path
    prepare_nifti_input "$subject/$hemi.topo" topo_path

    correct_raw="$subject/$hemi.correct.nii"
    if ! "${SCRIPTS_DIR}/ReplaceWhiteMatter" -t "${tissue_path}" -p "${topo_path}" -o "${correct_raw}"; then
      echo "[ERROR] ReplaceWhiteMatter failed for ${subject} ${hemi}." >&2
      exit 1
    fi
    check_exists "${correct_raw}"

    if ! "${SCRIPTS_DIR}/VTK2PLY" -i "$subject/$hemi.orig.vtk" -o "$subject/$hemi.white.vtk.ply"; then
      echo "[ERROR] VTK2PLY failed for ${subject} ${hemi}." >&2
      exit 1
    fi
    check_exists "$subject/$hemi.white.vtk.ply"

    if ! "${SCRIPTS_DIR}/ComputeCurvature" -i "$subject/$hemi.white.vtk.ply" -o "$subject/$hemi.InnerSurf.vtk" -c -d -s 60; then
      echo "[ERROR] ComputeCurvature failed for ${subject} ${hemi}." >&2
      exit 1
    fi
    check_exists "$subject/$hemi.InnerSurf.vtk"

    correct_sulci_raw="$subject/$hemi.correct.sulci.nii"
    if ! "${SCRIPTS_DIR}/GeodesicDistanceTransform" -i "${correct_raw}" -p 5 -o "${correct_sulci_raw}"; then
      echo "[ERROR] GeodesicDistanceTransform failed for ${subject} ${hemi}." >&2
      exit 1
    fi
    check_exists "${correct_sulci_raw}"

    strSkeletonImageHdr="$subject/$hemi.correct.sulci.skeleton.compact.hdr"
    strSkeletonImage="$subject/$hemi.correct.sulci.skeleton.nii.gz"
    strSkeletonImageMy="$subject/$hemi.correct.sulci.skeleton.my.nii.gz"
    strSulciImageHdr="$subject/$hemi.correct.sulci.compact.hdr"

    if ! fslchfiletype NIFTI_STD::PAIR "${correct_sulci_raw}" "${strSulciImageHdr}"; then
      echo "[ERROR] fslchfiletype failed for ${subject} ${hemi}." >&2
      exit 1
    fi
    check_exists "${strSulciImageHdr}"
    check_exists "${strSulciImageHdr%.hdr}.img"

    echo "[INFO] Subject ${subject} ${hemi}: skeletonization" >&2
    time_start="$(date +%s)"
    if ! "${SCRIPTS_DIR}/skel/bin/VipSkeleton" -i "${strSulciImageHdr}" -so "${strSkeletonImageHdr}" -v n -c n -sk s >/dev/null; then
      echo "[ERROR] VipSkeleton failed for ${subject} ${hemi}." >&2
      exit 1
    fi
    if ! fslswapdim "${strSkeletonImageHdr}" -x y z "${strSkeletonImage}"; then
      echo "[ERROR] fslswapdim failed for ${subject} ${hemi}." >&2
      exit 1
    fi
    if ! "${SCRIPTS_DIR}/3DSkeleton" -i "${correct_sulci_raw}" -o "${strSkeletonImageMy}"; then
      echo "[ERROR] 3DSkeleton failed for ${subject} ${hemi}." >&2
      exit 1
    fi
    time_stop="$(date +%s)"
    usedTime=$((time_stop - time_start))
    echo "[INFO] Subject ${subject} ${hemi}: skeleton done in ${usedTime}s" >&2

    check_exists "${strSkeletonImage}"

    recon_sulci_raw="$subject/$hemi.correct.ReconverSulci.nii"
    if ! "${SCRIPTS_DIR}/RecoverSulci" -i "${correct_raw}" -s "$subject/$hemi.correct.sulci.skeleton.nii.gz" -o "${recon_sulci_raw}"; then
      echo "[ERROR] RecoverSulci failed for ${subject} ${hemi}." >&2
      exit 1
    fi
    check_exists "${recon_sulci_raw}"

    if ! "${SCRIPTS_DIR}/OuterSurfaceReconstruction" -t "${recon_sulci_raw}" -i "$subject/$hemi.InnerSurf.vtk" -n 500 -d; then
      echo "[ERROR] OuterSurfaceReconstruction failed for ${subject} ${hemi}." >&2
      exit 1
    fi

    gzip_if_exists "${correct_raw}" "${correct_sulci_raw}" "${recon_sulci_raw}"
  done
done

echo "[INFO] Done" >&2
