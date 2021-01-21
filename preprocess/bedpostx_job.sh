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

#SBATCH -o logs/bedpostx-%j.out

module load python-uon/gcc6.3.0

cmd="./run_subject_bedpostx.py $1 $2"

echo $cmd
eval $cmd
