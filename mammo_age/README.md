# Mammo-AGE: Deep Learning Estimation of Breast Age from Mammograms

Age is a well-known and pivotal factor relating to organ functions and health in the human body. 
For breast, biological aging of breast tissue manifests specific physiological changes distinct from chronological aging. 
This study introduces a deep learning model to estimate the biological age of the breast using healthy mammograms.


### Example training  (or `sh /script/sbatch/train_example.sh`):
```bash
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
```

### Example predicting (or `sh /script/sbatch/predict_example.sh`):
```bash
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
```

The configs above are meant to specify exact implementation details and our experimental procedure
and may need to be adjusted to your specific use case.



## Acknowledgements
This work is based on the following repositories: 
[POEs](https://github.com/Li-Wanhua/POEs),
[Global-LocalTransformer](https://github.com/shengfly/global-local-transformer),
and [Mean-Variance Loss](https://openaccess.thecvf.com/content_cvpr_2018/html/Pan_Mean-Variance_Loss_for_CVPR_2018_paper.html)
