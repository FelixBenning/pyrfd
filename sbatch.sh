#!/bin/bash

#SBATCH --partition single 
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=60gb

poetry run python benchmarking/classification $1 $2
