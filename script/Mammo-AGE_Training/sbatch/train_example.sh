#!/bin/bash
#SBATCH --job-name=Mammo-AGE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH -p XXX
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=XXX
#SBATCH  --qos=XXX
#SBATCH --output=./logs/%x-%j.out


cd ./mammo_ages

dataset=Custom_dataset
BS=32
ImgSize=1024
results_dir="/Path/to/your/logs"  # change to your own directory
dataset_config="dataset_configs/mammo_dataset.yaml" # change to your own dataset config
csv_dir="/Path/to/data/base_csv_folder"  # change to your own directory
image_dir="/Path/to/data/base_image_folder"  # change to your own directory

method="Mammo-AGE"

for fold in 0 1 2 3 4
do
  for arch in resnet18 efficientnet_b0 convnext_tiny resnet18 densenet121 resnet50
  do
    echo "$dataset - $method $arch fold--$fold"
    python train.py \
    --model-method $method \
    --arch $arch \
    --batch-size $BS \
    --img-size $ImgSize \
    --dataset $dataset \
    --dataset-config $dataset_config \
    --csv-dir $csv_dir \
    --image-dir $image_dir \
    --fold $fold \
    --epochs 100 \
    --lr 1e-3 \
    --nblock 10 \
    --hidden_size 128 \
    --results-dir $results_dir

  done
done