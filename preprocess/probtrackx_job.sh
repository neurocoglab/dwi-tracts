#!/usr/bin/bash

#SBATCH --partition=imgq
# This specifies type of node job will use

#SBATCH --nodes=1
# This specifies job uses 1 node

#SBATCH --ntasks-per-node=1
# This specifies job only use 10 cores on the node

#SBATCH --mem=20g
# This specifies maximum memory use will be 20 gigabytes

#SBATCH --time=30:00:00
# This specifies job will last no longer than 30 hours

#SBATCH -o logs/probtrackx-%j.out

# Load relevant modules here

cmd="./run_probtrackx.py $1 $2 $3"

echo $cmd
eval $cmd
