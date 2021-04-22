#!/bin/bash

#SBATCH --job-name=rnn-alg
#SBATCH --account=mallet
#SBATCH --time=0-00:01:00
#SBATCH -o /home/zharris1/Documents/Jobs/RNN-NAS/out.txt
#SBATCH -e /home/zharris1/Documents/Jobs/RNN-NAS/err.txt
###


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

python "/home/zharris1/Documents/Github/RNN-NAS/lstm.py"