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
    hidden_dim=$2;
else
  echo "Hidden dim set to default to feature number (-1)"
fi

cmd="ovhai job run --name binaps-num-$1 --flavor ai1-1-gpu --gpu $nbr_gpu --volume output2@GRA/:/workspace/container_0:RW tchataing/binaps:run -- python /workspace/binaps_explore/Binaps_code/main.py --hidden_dim $hidden_dim --thread_num $(($nbr_gpu * 13)) --save_model --output_dir	/workspace/container_0 -i";

input_dir="/workspace/container_0/data/";
input_files=("github_cyber_2022-07-17T12h29m06s__1w_spaced_1d_smallest_6.0.dat" "github_cyber_2022-07-17T12h31m01s__1w_spaced_1d_smallest_6.0_50uni.dat" "github_cyber_2022-07-17T12h34m11s__1w_spaced_1d_smallest_6.0_50log.dat" "github_cyber_2022-07-17T12h50m45s__1w_spaced_1d_smallest_24.0.dat" "github_cyber_2022-07-17T12h52m36s__1w_spaced_1d_smallest_24.0_50uni.dat" "github_cyber_2022-07-17T12h55m50s__1w_spaced_1d_smallest_24.0_50log.dat" "github_cyber_2022-07-17T13h03m50s__1w_spaced_2d_smallest_6.0.dat" "github_cyber_2022-07-17T13h04m47s__1w_spaced_2d_smallest_6.0_50uni.dat" "github_cyber_2022-07-17T13h06m17s__1w_spaced_2d_smallest_6.0_50log.dat" "github_cyber_2022-07-17T13h13m59s__1w_spaced_2d_smallest_24.0.dat" "github_cyber_2022-07-17T13h14m49s__1w_spaced_2d_smallest_24.0_50uni.dat" "github_cyber_2022-07-17T13h16m19s__1w_spaced_2d_smallest_24.0_50log.dat" "github_cyber_2022-07-17T13h16m19s__1w_spaced_2d_smallest_24.0_50log.dat");

$cmd $input_dir${input_files[$1]};
