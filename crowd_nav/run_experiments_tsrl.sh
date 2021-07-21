#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
# Script to reproduce results
for ((i=0;i<10;i+=1))
do 
	python train.py \
	--policy tree-search-rl \
	--output_dir data/$day/tsrl/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/ts_separate.py

#	python train.py \
#	--policy model-predictive-rl \
#	--output_dir data/$day/mprl/$i \
#	--randomseed $i  \
#	--config configs/icra_benchmark/mp_separate.py
done

