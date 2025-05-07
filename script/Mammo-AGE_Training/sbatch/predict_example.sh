#!/bin/bash
#SBATCH --job-name=Predict-Mammo-AGE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH -p XXX
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=XXX
#SBATCH  --qos=XXX
#SBATCH --output=./logs/%x-%j.out

nvidia-smi

cd ./mammo_ages

method="Mammo-AGE"
base_dir='/Path/to/your/chickpoint/folder'

for fold in 0 1 2 3 4
do
  echo "fold--$fold"
  resultdir="$base_dir/1024_fold${fold}"
  echo "Result-Dir: $resultdir"

  for dataset in Custom_dataset
  do
    echo "$method -- Predict: $dataset -- fold--$fold"
    python predict.py --results-dir $resultdir --dataset $dataset

  done
done