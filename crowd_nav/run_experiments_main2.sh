#!/bin/bash
day=`date +%m%d`
a=0.01
b=-0.25
c=1.0
d=0.125
echo "The Script begin at $day"
# Script to reproduce results
for ((i=0;i<3;i+=1))
do 
	python train.py \
	--policy tree-search-rl \
	--output_dir data/$day/tsrl1/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/ts_separate.py \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--safe_weight $d \
	--human_num 5

	python train.py \
	--policy model-predictive-rl \
	--output_dir data/$day/mprl1/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/mp_separate.py \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--safe_weight $d \
	--human_num 5

	python train.py \
	--policy sarl \
	--output_dir data/$day/sarl1/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/sarl.py \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--safe_weight $d \
	--human_num 5
done
