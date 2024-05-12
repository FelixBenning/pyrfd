#!/bin/bash

#SBATCH --partition single 
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20gb

# poetry run python benchmarking/classification
poetry run python visualize_covariance_fit.py