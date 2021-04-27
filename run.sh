#!/bin/bash

#SBATCH --job-name=rnn-alg
#SBATCH --account=mallet
#SBATCH --time=0-04:00:00
#SBATCH -o /home/zharris1/Documents/Jobs/RNN-NAS/slurms/out_%A.txt
#SBATCH -e /home/zharris1/Documents/Jobs/RNN-NAS/slurms/err_%A.txt
###


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

python "/home/zharris1/Documents/Github/RNN-NAS/lstm.py"