#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $d1"
# Script to reproduce results
for ((i=0;i<3;i+=1))
do 
	python train.py \
	--policy tree-search-rl \
	--output_dir data/$day/tsrl/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/ts_separate.py
done
