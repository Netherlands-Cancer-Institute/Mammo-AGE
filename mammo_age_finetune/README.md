# Finetune Mammo-AGE for downstream tasks: Breast Cancer Risk Prediction and Diagnosis
Furthermore, we applied the model specifically for downstream breast cancer diagnosis and risk prediction.


### Example training (or `sh /script/sbatch/train_example.sh`):
```bash
cd ./mammo_age_finetune

BS=12
ImgSize=1536 # 256 512 1024 1536 2048
arch="resnet18" # resnet18, resnet50, efficientnet_b0, densenet121, convnext_tiny
base_model_path="/Path/to/your/mammo_age_logs/img_size_${ImgSize}/$arch"
results_dir="/Path/to/your/logs/finetune_${ImgSize}/"  # change to your own directory
csv_dir="/Path/to/data/base_csv_folder"  # change to your own directory
image_dir="/Path/to/data/base_image_folder"  # change to your own directory
dataset_config="dataset_configs/mammo_dataset.yaml" # change to your own dataset config
dataset=Custom_dataset
fraction=1.0 # 0.05 0.1 0.3 0.5 1.0

for fold in 0
do
  pretrained_model_path="$base_model_path/${ImgSize}_fold$fold/model_best.pth.tar"
  echo " $arch fold--$fold"
  python train_risk.py \
  --seed 42 \
  --task risk \
  --num-output-neurons 6 \
  --pretrained_model_path $pretrained_model_path \
  --model_method Finetune \
  --dataset $dataset \
  --dataset-config $dataset_config \
  --csv-dir $csv_dir \
  --image-dir $image_dir \
  --max_followup 5 \
  --years_at_least_followup 3 \
  --accumulation_steps 24 \
  --batch-size $BS \
  --img-size $ImgSize \
  --results-dir $results_dir \
  --fraction $fraction \

done
```

### Example predicting (or `sh /script/sbatch/predict_example.sh`):
```bash
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
```

The configs above are meant to specify exact implementation details and our experimental procedure
and may need to be adjusted to your specific use case.