#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
# Script to reproduce results
for ((i=0;i<3;i+=1))
do 
	python train.py \
	--policy td3_rl \
	--output_dir data/$day/td3/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/td3.py
done
