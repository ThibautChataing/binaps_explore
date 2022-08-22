#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied";
    exit 1;
fi

nbr_gpu=1;
if [ "$2" ]
  then
    nbr_gpu=$2;
else
  echo "Number of GPU set to default 1"
fi

hidden_dim=-1;
if [ "$3" ]
  then
    hidden_dim=$3;
else
  echo "Hidden dim set to default to feature number (-1)"
fi

no_gpu=""
if [ "$4" = '--no_gpu' ]
  then
    no_gpu=$4;
    echo "CPU only"
fi

cmd="ovhai job run --name binaps-num-$1 --flavor ai1-1-gpu --gpu $nbr_gpu --volume out@GRA/:/workspace/container_0:RW tchataing/binaps:run -- python /workspace/binaps_explore/Binaps_code/main.py --hidden_dim $hidden_dim --thread_num $(($nbr_gpu * 13)) --save_model $no_gpu --output_dir	/workspace/container_0 -i";

input_dir="/workspace/container_0/";
input_files=("github_cyber_2022-07-22T16h07m49s__1w_spaced_1d_smallest_6.0_50uni.dat" "github_cyber_2022-07-22T16h10m55s__1w_spaced_1d_smallest_6.0_50log.dat" "github_cyber_2022-07-22T16h12m44s__1w_spaced_1d_smallest_24.0_50uni.dat" "github_cyber_2022-07-22T16h15m46s__1w_spaced_1d_smallest_24.0_50log.dat" "github_cyber_2022-07-22T16h16m37s__1w_spaced_2d_smallest_6.0_50uni.dat" "github_cyber_2022-07-22T16h18m09s__1w_spaced_2d_smallest_6.0_50log.dat" "github_cyber_2022-07-22T16h19m01s__1w_spaced_2d_smallest_24.0_50uni.dat" "github_cyber_2022-07-22T16h20m33s__1w_spaced_2d_smallest_24.0_50log.dat"
);

$cmd $input_dir${input_files[$1]};
