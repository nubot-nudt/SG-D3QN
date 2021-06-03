#!/bin/bash

# Script to reproduce results
for ((i=0;i<10;i+=1))
do 
	python train.py \
	--policy tree-search-rl \
	--output_dir data/0603/$i
	--randomseed $i
done
