#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
a=0.1
b=-0.25
c=1.0
# Script to reproduce results
for ((i=0;i<1;i+=1))
do
	for((j=1;j<5;j+=1))
	do	
		for((k=1;k<3;k+=1))
		do	
			for((m=1;m<5;m+=1))
			do	
				python train.py \
				--policy tree-search-rl \
				--output_dir data/$day/tsrl/$i \
				--randomseed $i  \
				--config configs/icra_benchmark/ts_separate.py \
				--safe_weight 1.0 \
				--goal_weight `echo " $a * $j" | bc` \
				--re_collision `echo " $b * $k" | bc` \
				--re_arrival `echo " $c * $m" | bc`
			done
		done
	done
#	python train.py \
#	--policy model-predictive-rl \
#	--output_dir data/$day/mprl/$i \
#	--randomseed $i  \
#	--config configs/icra_benchmark/mp_separate.py
done

