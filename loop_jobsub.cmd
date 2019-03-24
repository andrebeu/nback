#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")
wd_dir="/tigress/abeukers/wd/epm"


## now loop through the above array
for i in {1..25}; do 
	for EDIM in {1..10}; do 
		sbatch ${wd_dir}/gpu_jobsub.cmd "${EDIM}" 
	done
done
