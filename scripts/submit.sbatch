#!/bin/bash

#SBATCH --account=pr_341_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=3:30:00
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|a100'
#SBATCH --job-name=qsvd_llavanext
#SBATCH --output=qsvd_llavanext-%j-%a.out # set the output file name for each task. The %j is replaced by the job ID and %a is replaced by the task ID.


module purge

bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/runnext.sh