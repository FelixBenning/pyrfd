#!/bin/bash

#SBATCH --partition single 
#SBATCH --ntasks=1
#SBATCH --time=00:02:00
#SBATCH --gres=gpu:1

poetry run python test.py
