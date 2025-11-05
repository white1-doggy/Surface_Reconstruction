#!/bin/bash
#SBATCH --job-name=pial_surface
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --partition=bme_cpu
#SBATCH -t 240:00:00

# Script for FreeSurfer Step by Step
# Load FreeSurfer and Specify File Root
# Specify subject direction

module load compiler/gcc/7.3.1
SCRIPTS_DIR='/public_bme/home/liujm/Code/MD_InfanTSurf/Scripts/Step8_Surf_Recons/'
file_root="/home_data/home/lianzf2024/test2/Step02_sMRI_Surf"

cd $file_root
echo $file_root

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
    for hemi in lh rh ; do
      # 使用矫正之后的WM替代原来的分割
      $SCRIPTS_DIR'/'ReplaceWhiteMatter -t $i/$hemi'.nii.gz' -p $i/$hemi'.topo.nii.gz' -o $i/$hemi'.correct.nii.gz';
      # 将vtk格式转成ply格式的3D (pointcloud也可以设置成这个格式)
      $SCRIPTS_DIR'/'VTK2PLY -i $i/$hemi'.orig.vtk' -o $i/$hemi'.white.vtk.ply'
      # 计算内皮层的曲率
      $SCRIPTS_DIR'/'ComputeCurvature -i $i/$hemi'.white.vtk.ply' -o $i/$hemi'.InnerSurf.vtk' -c -d -s 60;
      #
      $SCRIPTS_DIR'/'GeodesicDistanceTransform -i $i/$hemi'.correct.nii.gz' -p 5 -o $i/$hemi'.correct.sulci.nii.gz';
      # 计算白质的skeleton
      strSkeletonImageHdr=$i/$hemi'.correct.sulci.skeleton.compact.hdr';
      strSkeletonImage=$i/$hemi'.correct.sulci.skeleton.nii.gz';
      strSkeletonImageMy=$i/$hemi'.correct.sulci.skeleton.my.nii.gz';
      strSulciImageHdr=$i/$hemi'.correct.sulci.compact.hdr';
      fslchfiletype NIFTI_STD::PAIR $i/$hemi'.correct.sulci.nii.gz' ${strSulciImageHdr}

      echo "Doing skeletonization...";
      time_start="$(date +%s)"
      $SCRIPTS_DIR'/skel/bin/VipSkeleton' -i ${strSulciImageHdr} -so ${strSkeletonImageHdr} -v n -c n -sk s > /dev/null;
      fslswapdim ${strSkeletonImageHdr} -x y z ${strSkeletonImage}
      $SCRIPTS_DIR'/'3DSkeleton -i $i/$hemi'.correct.sulci.nii.gz' -o ${strSkeletonImageMy};
      time_stop="$(date +%s)"
      usedTime=`expr ${time_stop} - ${time_start}`;
      echo "Skeleton done! Cost ${usedTime} seconds.";
      # 脑沟refine
      $SCRIPTS_DIR'/'RecoverSulci -i $i/$hemi'.correct.nii.gz' -s $i/$hemi'.correct.sulci.skeleton.nii.gz' -o $i/$hemi'.correct.ReconverSulci.nii.gz';
      # 重建外皮层大概需要一个小时，起始的是经过修复后的内皮层
      $SCRIPTS_DIR'/'OuterSurfaceReconstruction -t $i/$hemi'.correct.ReconverSulci.nii.gz' -i $i/$hemi'.InnerSurf.vtk' -n 500 -d
    done
    echo >&6
  } &
done
wait

echo "Done"

