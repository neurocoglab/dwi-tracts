#!/usr/bin/bash

#SBATCH --partition=imgcomputeq
# This specifies type of node job will use

#SBATCH --nodes=1
# This specifies job uses 1 node

#SBATCH --ntasks-per-node=1
# This specifies job only use 10 cores on the node

#SBATCH --mem=10g
# This specifies maximum memory use will be 10 gigabytes

#SBATCH --time=02:00:00
# This specifies job will last no longer than 2 hours

#SBATCH -o logs/bedpostx_preproc-%j.out

#SBATCH --qos=img

#SBATCH --export=NONE

# Load relevant modules here

# Preprocessing steps
cmd="./run_bedpostx_preproc.py $1 $2"

echo $cmd
eval $cmd
