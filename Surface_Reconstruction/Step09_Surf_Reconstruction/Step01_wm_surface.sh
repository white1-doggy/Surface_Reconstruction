#!/bin/bash
#SBATCH --job-name=wm_surface
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --partition=bme_cpu
#SBATCH -t 240:00:00

# Script for FreeSurfer Step by Step
# Load FreeSurfer and Specify File Root

# Specify subject direction
module load compiler/gcc/7.3.1

file_root="/home_data/home/lianzf2024/test2/Step02_sMRI_Surf"

cd $file_root

Thread_num=56
#命名管道文件
Tmp_fifo=/tmp/$$.fifo
#创建命名管道文件
mkfifo $Tmp_fifo
#用文件句柄(随便给个6)打开管道文件
exec 6<> $Tmp_fifo
rm -f $Tmp_fifo

for i in `seq $Thread_num`
do
        #向管道中放入最大并发数个行，供下面read读取
        echo >&6
done

for i in `ls $file_root`
do
  read -u 6
  {
    fullname=$(basename $i .nii.gz)
    # Step 1-5
  /public_bme/home/liujm/Code/MD_InfanTSurf/Scripts/Step8_Surf_Recons/TopologyCorrectionLevelSet --tissue $i/"lh.nii.gz" --out $i/"lh.topo.nii.gz"  -r -d 10
  /public_bme/home/liujm/Code/MD_InfanTSurf/Scripts/Step8_Surf_Recons/IsoSurface -t $i/"lh.topo.nii.gz" -o $i/"lh.orig.vtk"

  /public_bme/home/liujm/Code/MD_InfanTSurf/Scripts/Step8_Surf_Recons/TopologyCorrectionLevelSet --tissue $i/"rh.nii.gz" --out $i/"rh.topo.nii.gz"  -r -d 10
  /public_bme/home/liujm/Code/MD_InfanTSurf/Scripts/Step8_Surf_Recons/IsoSurface -t $i/"rh.topo.nii.gz" -o $i/"rh.orig.vtk"

    echo >&6
  } &
done
wait

echo "Done"

