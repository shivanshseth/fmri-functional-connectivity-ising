#!/bin/bash
#SBATCH -n 35
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mail-user=shivansh.seth@research.iiit.ac.in


~/miniconda3/envs/brain/bin/python -u abide_dataset.py > dataset.log 2>&1
