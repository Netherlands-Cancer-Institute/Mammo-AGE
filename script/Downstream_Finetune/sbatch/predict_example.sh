#!/bin/bash
#SBATCH --job-name=Mammo-AGE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH -p XXX
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=XXX
#SBATCH  --qos=XXX
#SBATCH --output=./logs_finetune/%x-%j.out

nvidia-smi

cd ./mammo_ages

cd ./mammo_age_finetune

BS=1
ImgSize=1536 # 256 512 1024 1536 2048
arch="resnet18" # resnet18, resnet50, efficientnet_b0, densenet121, convnext_tiny
mode="Mammo-AGE" #
task="risk"
base_results_dir="/Path/to/your/logs/finetune_${ImgSize}/${task}/Custom_dataset"  # change to your own directory
csv_dir="/Path/to/data/base_csv_folder"  # change to your own directory
image_dir="/Path/to/data/base_image_folder"  # change to your own directory
dataset_config="dataset_configs/mammo_dataset.yaml" # change to your own dataset config
dataset=Custom_dataset

for fold in 0
do
  echo " $arch fold--$fold"

  results_dir="${base_results_dir}/${mode}-${arch}/${ImgSize}_fold${fold}"

  python predict.py \
  --seed 42 \
  --task risk \
  --num-output-neurons 6 \
  --dataset $dataset \
  --dataset-config $dataset_config \
  --csv-dir $csv_dir \
  --image-dir $image_dir \
  --max_followup 5 \
  --years_at_least_followup 3 \
  --batch-size $BS \
  --img-size $ImgSize \
  --results-dir $results_dir

done