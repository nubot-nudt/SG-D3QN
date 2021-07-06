#!/bin/bash

# Script to reproduce results
for ((i=0;i<3;i+=1))
do 
	python train.py \
	--policy model-predictive-rl \
	--output_dir data/0704/mprl/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/mp_separate.py
done
