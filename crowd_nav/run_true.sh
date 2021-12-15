#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
# Script to reproduce results


for ((i=0;i<4;i+=1))
do

	python test.py \
	--model_dir data/visible/true1/5human_visible_holonomic/$i \
	--human_num 5 \
	--planning_depth 2
done
