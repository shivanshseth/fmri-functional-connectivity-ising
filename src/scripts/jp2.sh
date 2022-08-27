#!/bin/bash
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mail-user=shivansh.seth@research.iiit.ac.in

~/miniconda3/envs/brain/bin/python -u classifier.py > cgd_classifier.log

