#!/bin/bash
#SBATCH -q debug
#SBATCH -t 00:02:00
#SBATCH -N 1
#SBATCH -C knl
#BB create_persistent name=DeepCAM capacity=450GB access_mode=striped type=scratch
