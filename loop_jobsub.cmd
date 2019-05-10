#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/nback"
##

declare -a nback_arr=(1 2 3)
declare -a ntokens_arr=(3 4 5 10)

for seed in {5..25}; do 
	for nback in "${nback_arr[@]}"; do 
		for ntokens in "${ntokens_arr[@]}"; do 
			sbatch ${wd_dir}/gpu_jobsub.cmd "${seed}" "${nback}" "${ntokens}"
		done
	done
done
