#!/bin/bash

printf "\n\n -- cluster_loop\n"

stamp=$(date +"%m%d%H%M%S")

##
wd_dir="/tigress/abeukers/wd/nback"
##

declare -a setsize_arr=(3 4 5 6)
declare -a ntrials_arr=(2 3 4 5 10 15 20)

for seed in {0..3}; do
  for emthresh in {0..10}; do 
  	for setsize in "${setsize_arr[@]}"; do 
  		for ntrials in "${ntrials_arr[@]}"; do 
  			sbatch ${wd_dir}/gpu_jobsub.cmd ${seed} ${ntrials} ${setsize} ${emthresh}
      done
		done
	done
done
