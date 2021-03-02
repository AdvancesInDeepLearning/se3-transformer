#!/bin/bash
echo $PWD
for DATASETNAME in WaterDrop Water Sand Goop MultiMaterial RandomFloor WaterRamps SandRamps FluidShake FluidShakeBox Continuous WaterDrop-XL Water-3D Sand-3D Goop-3D
do
        $PWD/download_dataset.sh $DATASETNAME learning-to-simulate
done
