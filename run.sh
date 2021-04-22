#!/bin/bash

#SBATCH --job-name=rnn-alg
#SBATCH --account=mallet
#SBATCH --time=0-00:01:00
###


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

python lstm.py