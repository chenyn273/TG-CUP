#!/bin/bash
#SBATCH -p x86_64_GPU
#SBATCH -n 1
#SBATCH -G 2
#SBATCH -o job.out
python main.py
