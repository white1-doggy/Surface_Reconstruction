#!/bin/bash
#SBATCH --job-name=Surface_Metric
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --partition=bme_cpu
#SBATCH -t 240:00:00

module load compiler/gcc/7.3.1
export FREESURFER_HOME=/public_bme/software/freesurfer7.2/freesurfer/
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/home_data/home/lianzf2024/test2/Step03_sMRI_Metric
iBEAT_results_DIR=/home_data/home/lianzf2024/test2/Step02_sMRI_Surf
SCRIPTS_DIR=/public_bme/home/liujm/Code/MD_InfanTSurf/Scripts/Step8_Surf_Recons/
echo $iBEAT_results_DIR


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


for SUBJ_ID in `ls $iBEAT_results_DIR`
do
  read -u 6
  {
      mkdir $SUBJECTS_DIR'/'$SUBJ_ID
      cp $iBEAT_results_DIR'/'$SUBJ_ID'/aseg.nii.gz' $SUBJECTS_DIR'/'$SUBJ_ID'/'
      cp $iBEAT_results_DIR'/'$SUBJ_ID'/mprage.nii.gz' $SUBJECTS_DIR'/'$SUBJ_ID'/'
      cp $iBEAT_results_DIR'/'$SUBJ_ID'/masked.nii.gz' $SUBJECTS_DIR'/'$SUBJ_ID'/'
      cp $iBEAT_results_DIR'/'$SUBJ_ID'/lh.correct.nii.gz' $SUBJECTS_DIR'/'$SUBJ_ID'/'
      cp $iBEAT_results_DIR'/'$SUBJ_ID'/rh.correct.nii.gz' $SUBJECTS_DIR'/'$SUBJ_ID'/'

      cp $iBEAT_results_DIR'/'$SUBJ_ID'/lh.InnerSurf.vtk.InnerSurf.vtk' $SUBJECTS_DIR'/'$SUBJ_ID'/lh.InnerSurf.vtk'
      cp $iBEAT_results_DIR'/'$SUBJ_ID'/lh.InnerSurf.vtk.MiddleSurf.vtk' $SUBJECTS_DIR'/'$SUBJ_ID'/lh.MiddleSurf.vtk'
      cp $iBEAT_results_DIR'/'$SUBJ_ID'/lh.InnerSurf.vtk.OuterSurf.vtk' $SUBJECTS_DIR'/'$SUBJ_ID'/lh.OuterSurf.vtk'

      cp $iBEAT_results_DIR'/'$SUBJ_ID'/rh.InnerSurf.vtk.InnerSurf.vtk' $SUBJECTS_DIR'/'$SUBJ_ID'/rh.InnerSurf.vtk'
      cp $iBEAT_results_DIR'/'$SUBJ_ID'/rh.InnerSurf.vtk.MiddleSurf.vtk' $SUBJECTS_DIR'/'$SUBJ_ID'/rh.MiddleSurf.vtk'
      cp $iBEAT_results_DIR'/'$SUBJ_ID'/rh.InnerSurf.vtk.OuterSurf.vtk' $SUBJECTS_DIR'/'$SUBJ_ID'/rh.OuterSurf.vtk'

      cd $SUBJECTS_DIR'/'$SUBJ_ID
      mkdir ./surf # surface
      mkdir ./label # label

      # 转化到workbench的surface
      for hemi in lh rh ; do
        # 第一步做空间转化 （内皮层，中间皮层，外皮层），指的是在workbench中对应上 xxx.PhysicalSpace (workbench)
        $SCRIPTS_DIR'/'SurfaceToPhysicalSpace -s $hemi'.InnerSurf.vtk' -i $hemi'.correct.nii.gz' -o $hemi'.InnerSurf.PhysicalSpace.vtk' -t 1
        $SCRIPTS_DIR'/'SurfaceToPhysicalSpace -s $hemi'.MiddleSurf.vtk' -i $hemi'.correct.nii.gz' -o $hemi'.MiddleSurf.PhysicalSpace.vtk' -t 1
        $SCRIPTS_DIR'/'SurfaceToPhysicalSpace -s $hemi'.OuterSurf.vtk' -i $hemi'.correct.nii.gz' -o $hemi'.OuterSurf.PhysicalSpace.vtk' -t 1

        # 将workbench的PhysicalSpace转化成workbench可以使用的gifti数据
        mris_convert $hemi'.InnerSurf.PhysicalSpace.vtk' $hemi'.white.surf.gii'
        mris_convert $hemi'.MiddleSurf.PhysicalSpace.vtk' $hemi'.midthickness.surf.gii'
        mris_convert $hemi'.OuterSurf.PhysicalSpace.vtk' $hemi'.pial.surf.gii'

        # 使用workbench将workbench空间转化到freesurfer空间
        wb_command -surface-apply-affine $hemi'.white.surf.gii' $SCRIPTS_DIR'/toFS.mat' $hemi'.white.FS.surf.gii'
        mris_convert $hemi'.white.FS.surf.gii' $hemi'.white'

        wb_command -surface-apply-affine $hemi'.midthickness.surf.gii' $SCRIPTS_DIR'/toFS.mat' $hemi'.midthickness.FS.surf.gii'
        mris_convert $hemi'.midthickness.FS.surf.gii' $hemi'.midthickness'

        wb_command -surface-apply-affine $hemi'.pial.surf.gii' $SCRIPTS_DIR'/toFS.mat' $hemi'.pial.FS.surf.gii'
        mris_convert $hemi'.pial.FS.surf.gii' $hemi'.pial'

        # 删除中间变量
        rm ./$hemi'.InnerSurf.vtk'
        rm ./$hemi'.MiddleSurf.vtk'
        rm ./$hemi'.OuterSurf.vtk'
        rm ./$hemi'.InnerSurf.PhysicalSpace.vtk'
        rm ./$hemi'.MiddleSurf.PhysicalSpace.vtk'
        rm ./$hemi'.OuterSurf.PhysicalSpace.vtk'
        rm ./$hemi'.white.FS.surf.gii'
        rm ./$hemi'.midthickness.FS.surf.gii'
        rm ./$hemi'.pial.FS.surf.gii'

        mv ./$hemi'.white' ./surf/$hemi'.white'
        mv ./$hemi'.midthickness' ./surf/$hemi'.midthickness'
        mv ./$hemi'.pial' ./surf/$hemi'.pial'
      done

      # 在Freesurfer空间上计算一些参数
      cd $SUBJECTS_DIR'/'$SUBJ_ID'/surf/'
      # 在计算之前需要先smooth一下
      for hemi in lh rh ; do
        # smooth 并且计算内皮层的表面积
        mris_smooth -n 3 $hemi.white $hemi.smoothwm
        # from surface to sphere
        mris_inflate $hemi.smoothwm $hemi.inflated
        # 计算curvature （freesurfer and HCP）
        mris_curvature -thresh .999 -n -a 5 -w -distances 10 10 $hemi.inflated
        # sphere calculate
        mris_sphere $hemi.inflated $hemi.sphere
        # 球面的配准
        mris_register -curv $hemi.sphere $FREESURFER_HOME/average/$hemi.average.curvature.filled.buckner40.tif $hemi.sphere.reg
        mris_ca_label $SUBJ_ID $hemi $hemi.sphere.reg $FREESURFER_HOME/average/$hemi.curvature.buckner40.filled.desikan_killiany.2010-03-25.gcs $hemi.aparc.annot
        mris_ca_label $SUBJ_ID $hemi $hemi.sphere.reg $FREESURFER_HOME/average/$hemi.destrieux.simple.2009-07-29.gcs $hemi.aparc.a2009s.annot

        # 计算thickness
        mris_thickness $SUBJ_ID $hemi $hemi.thickness
        # 计算外皮层的表面积
        mris_smooth -n 0 -c pial.curv -b pial.area $hemi.pial $hemi.fakesmooth.pial
        rm $hemi.fakesmooth.pial
      done

      # 最终的格式转换
      cd $SUBJECTS_DIR'/'$SUBJ_ID
      # workbench 参数map转换
      wb_command -add-to-spec-file ./$SUBJ_ID.wb.spec INVALID ./mprage.nii.gz
      wb_command -set-structure ./lh.white.surf.gii CORTEX_LEFT -surface-type ANATOMICAL -surface-secondary-type GRAY_WHITE
      wb_command -add-to-spec-file ./$SUBJ_ID.wb.spec CORTEX_LEFT ./lh.white.surf.gii
      wb_command -set-structure ./lh.pial.surf.gii CORTEX_LEFT -surface-type ANATOMICAL -surface-secondary-type PIAL
      wb_command -add-to-spec-file ./$SUBJ_ID.wb.spec CORTEX_LEFT ./lh.pial.surf.gii
      wb_command -set-structure ./lh.midthickness.surf.gii CORTEX_LEFT -surface-type ANATOMICAL -surface-secondary-type MIDTHICKNESS
      wb_command -add-to-spec-file ./$SUBJ_ID.wb.spec CORTEX_LEFT ./lh.midthickness.surf.gii
      wb_command -set-structure ./rh.white.surf.gii CORTEX_RIGHT -surface-type ANATOMICAL -surface-secondary-type GRAY_WHITE
      wb_command -add-to-spec-file ./$SUBJ_ID.wb.spec CORTEX_RIGHT ./rh.white.surf.gii
      wb_command -set-structure ./rh.pial.surf.gii CORTEX_RIGHT -surface-type ANATOMICAL -surface-secondary-type PIAL
      wb_command -add-to-spec-file ./$SUBJ_ID.wb.spec CORTEX_RIGHT ./rh.pial.surf.gii
      wb_command -set-structure ./rh.midthickness.surf.gii CORTEX_RIGHT -surface-type ANATOMICAL -surface-secondary-type MIDTHICKNESS
      wb_command -add-to-spec-file ./$SUBJ_ID.wb.spec CORTEX_RIGHT ./rh.midthickness.surf.gii

      for Hemisphere in L R ; do
          #Set a bunch of different ways of saying left and right
          if [ $Hemisphere = "L" ] ; then
              hemisphere="l"
              Structure="CORTEX_LEFT"
          elif [ $Hemisphere = "R" ] ; then
              hemisphere="r"
              Structure="CORTEX_RIGHT"
          fi

        for Map in sulc@sulc@Sulc curv@curvature@Curvature ; do
            fsname=$(echo $Map | cut -d "@" -f 1)
            wbname=$(echo $Map | cut -d "@" -f 2)
            mapname=$(echo $Map | cut -d "@" -f 3)
            mris_convert -c ./surf/"$hemisphere"h."$fsname" "$hemisphere"h.white.surf.gii ./"$SUBJ_ID"."$Hemisphere"."$wbname".shape.gii
            wb_command -set-structure ./"$SUBJ_ID"."$Hemisphere"."$wbname".shape.gii ${Structure}
            wb_command -set-map-names ./"$SUBJ_ID"."$Hemisphere"."$wbname".shape.gii -map 1 "$SUBJ_ID"_"$Hemisphere"_"$mapname"
            wb_command -metric-palette ./"$SUBJ_ID"."$Hemisphere"."$wbname".shape.gii MODE_AUTO_SCALE_PERCENTAGE -pos-percent 2 98 -palette-name cool-warm -disp-pos true -disp-neg true -disp-zero true
          done

        for Map in thickness@thickness@thickness ; do
            fsname=$(echo $Map | cut -d "@" -f 1)
            wbname=$(echo $Map | cut -d "@" -f 2)
            mapname=$(echo $Map | cut -d "@" -f 3)
            mris_convert -c ./surf/"$hemisphere"h."$fsname" "$hemisphere"h.white.surf.gii ./"$SUBJ_ID"."$Hemisphere"."$wbname".shape.gii
            wb_command -set-structure ./"$SUBJ_ID"."$Hemisphere"."$wbname".shape.gii ${Structure}
            wb_command -set-map-names ./"$SUBJ_ID"."$Hemisphere"."$wbname".shape.gii -map 1 "$SUBJ_ID"_"$Hemisphere"_"$mapname"
            wb_command -metric-palette ./"$SUBJ_ID"."$Hemisphere"."$wbname".shape.gii MODE_AUTO_SCALE_PERCENTAGE -pos-percent 4 96 -interpolate true -palette-name videen_style -disp-pos true -disp-neg false -disp-zero false
        done

        for Map in aparc aparc.a2009s ; do #Remove BA because it doesn't convert properly
            if [ -e ./label/"$hemisphere"h."$Map".annot ] ; then
                mris_convert --annot ./label/"$hemisphere"h."$Map".annot "$hemisphere"h.white.surf.gii ./"$SUBJ_ID"."$Hemisphere"."$Map".label.gii
                wb_command -set-structure ./"$SUBJ_ID"."$Hemisphere"."$Map".label.gii $Structure
                wb_command -set-map-names ./"$SUBJ_ID"."$Hemisphere"."$Map".label.gii -map 1 "$SUBJ_ID"_"$Hemisphere"_"$Map"
                wb_command -gifti-label-add-prefix ./"$SUBJ_ID"."$Hemisphere"."$Map".label.gii "${Hemisphere}_" ./"$SUBJ_ID"."$Hemisphere"."$Map".label.gii
            fi
        done

      done

      wb_command -cifti-create-dense-scalar ./"$SUBJ_ID".sulc.dscalar.nii -left-metric ./"$SUBJ_ID".L.sulc.shape.gii -right-metric ./"$SUBJ_ID".R.sulc.shape.gii
      wb_command -set-map-names ./"$SUBJ_ID".sulc.dscalar.nii -map 1 "${SUBJ_ID}_Sulc"
      wb_command -cifti-palette ./"$SUBJ_ID".sulc.dscalar.nii MODE_AUTO_SCALE_PERCENTAGE ./"$SUBJ_ID".sulc.dscalar.nii -pos-percent 2 98 -palette-name cool-warm -disp-pos true -disp-neg true -disp-zero true
      wb_command -add-to-spec-file ./"$SUBJ_ID".wb.spec INVALID ./"$SUBJ_ID".sulc.dscalar.nii

      wb_command -cifti-create-dense-scalar ./"$SUBJ_ID".curvature.dscalar.nii -left-metric ./"$SUBJ_ID".L.curvature.shape.gii -right-metric ./"$SUBJ_ID".R.curvature.shape.gii
      wb_command -set-map-names ./"$SUBJ_ID".curvature.dscalar.nii -map 1 "${SUBJ_ID}_Curvature"
      wb_command -cifti-palette ./"$SUBJ_ID".curvature.dscalar.nii MODE_AUTO_SCALE_PERCENTAGE ./"$SUBJ_ID".curvature.dscalar.nii -pos-percent 2 98 -palette-name cool-warm -disp-pos true -disp-neg true -disp-zero true
      wb_command -add-to-spec-file ./"$SUBJ_ID".wb.spec INVALID ./"$SUBJ_ID".curvature.dscalar.nii

      wb_command -cifti-create-dense-scalar ./"$SUBJ_ID".thickness.dscalar.nii -left-metric ./"$SUBJ_ID".L.thickness.shape.gii -right-metric ./"$SUBJ_ID".R.thickness.shape.gii
      wb_command -set-map-names ./"$SUBJ_ID".thickness.dscalar.nii -map 1 "${SUBJ_ID}_thickness"
      wb_command -cifti-palette ./"$SUBJ_ID".thickness.dscalar.nii MODE_AUTO_SCALE_PERCENTAGE ./"$SUBJ_ID".curvature.dscalar.nii -pos-percent 4 96 -interpolate true -palette-name videen_style -disp-pos true -disp-neg false -disp-zero false
      wb_command -add-to-spec-file ./"$SUBJ_ID".wb.spec INVALID ./"$SUBJ_ID".thickness.dscalar.nii
    echo >&6
  } &
done
wait

echo "Done"

