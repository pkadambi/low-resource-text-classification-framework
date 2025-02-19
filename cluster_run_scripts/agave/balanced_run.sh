#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -p gpu
#SBATCH -q wildfire
#SBATCH -t 0-10:00
#SBATCH -o output_balanced.1.out
#SBATCH -e output_balanced.1.err
#SBATCH --mail-type=ALL
#SBATCH --export=None
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-user=pkadambi@asu.edu
#SBATCH  -C V100
nvidia-smi
echo "SCRIPT STARTED ***********************"
while getopts d:l:m:s:e:y: option
do 
	case "${option}"
		in
		d)dataset=${OPTARG};;
		l)label_id=${OPTARG};;
		m)n_mc=${OPTARG};;
		s)query_step=${OPTARG};;
		e)nseed=${OPTARG};;
		y)nquery_steps=${OPTARG};;
	esac
done
			
echo "Runing dataset : $dataset"
echo "Using Label Columns : $label_id"
echo "Num MC Iters : $n_mc"
echo "Query Step : $query_step"
echo "N_Seed: $nseed"
echo "Number of Active Queries per MC Iteration : $nquery_steps"


source activate /home/pkadambi/mambaforge/envs/al_ibm
which python
cd /scratch/pkadambi/low-resource-text-classification-framework/
pwd
python ./experiment_runner_balanced.py \
	--dataset=$dataset\
	--label_id=$label_id\
	--n_mc=$n_mc\
	--query_step=$query_step\
	--nseed=$nseed\
	--nquery_steps=$nquery_steps\

