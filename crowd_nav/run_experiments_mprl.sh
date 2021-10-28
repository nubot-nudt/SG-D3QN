#!/bin/bash
day=`date +%m%d`
a=0.1
b=-2.5
c=10.0
d=0.125
echo "The Script begin at $day"
# Script to reproduce results
for ((i=0;i<3;i+=1))
do 
	python train.py \
	--policy model-predictive-rl \
	--output_dir data/$day/mprl/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/mp_separate.py \
	--safe_weight $d \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--human_num 5
done
