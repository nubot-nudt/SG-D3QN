#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
# Script to reproduce results

for ((i=4;i<5;i+=1))
do

	python test.py \
	--model_dir data/visible/false1/5human_visible_holonomic/$i \
	--human_num 5 \
	--planning_depth 2
done

#for ((i=0;i<4;i+=1))
#do

#	python test.py \
#	--model_dir data/visible/false1/10human_visible_holonomic/$i \
#	--human_num 10 \
#	--planning_depth 2
#done

