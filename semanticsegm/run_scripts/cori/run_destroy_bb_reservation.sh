#!/bin/bash
#SBATCH -q debug
#SBATCH -t 00:02:00
#SBATCH -N 1
#SBATCH -C knl
#BB destroy_persistent name=DeepCAM
