import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def dataloador(myDataset, data_infos, dataset_names, args, train=False):
    """
    Create a DataLoader for the given dataset.

    Args:
        myDataset: Custom dataset class.
        data_infos: List of DataFrames containing image paths and metadata.
        dataset_names: List of dataset names.
        args: Arguments containing image size and other parameters.
        train: Boolean indicating whether to use training transformations.

    Returns:
        DataLoader: DataLoader for the dataset.
    """

    # Define augmentations for training
    augments = [
        A.Resize(args.img_size, args.img_size // 2, interpolation=cv2.INTER_LINEAR),  # Resize images
        A.HorizontalFlip(p=0.5),  # Randomly flip images horizontally
        A.VerticalFlip(p=0.5),  # Randomly flip images vertically
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),  # Random brightness and contrast adjustments
        A.ShiftScaleRotate(rotate_limit=(-5, 5), p=0.3),  # Random shift, scale, and rotation
        A.OneOf([
            A.MedianBlur(blur_limit=3, p=0.3),  # Apply median blur
            A.GaussianBlur(blur_limit=3, p=0.3),  # Apply Gaussian blur
            A.GaussNoise(var_limit=(3.0, 9.0), p=0.3),  # Add Gaussian noise
        ], p=0.3),  # Apply one of the above blurs/noise
        A.OneOf([
            A.ElasticTransform(p=0.3),  # Apply elastic transformation
            A.GridDistortion(num_steps=5, distort_limit=1., p=0.3),  # Apply grid distortion
            A.OpticalDistortion(distort_limit=1., p=0.3),  # Apply optical distortion
        ], p=0.3),  # Apply one of the above distortions
        A.CoarseDropout(max_height=int(args.img_size * 0.1), max_width=int(args.img_size // 2 * 0.1), max_holes=8, p=0.2),  # Apply coarse dropout
        A.Normalize(mean=0.5, std=0.5),  # Normalize images
    ]
    augments.append(ToTensorV2())
    train_transform = A.Compose(augments)

    # Define transformations for testing
    test_transform = A.Compose([
        A.Resize(args.img_size, args.img_size // 2, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ])

    # Select transformations based on the training flag
    if train:
        transform, shuffle, drop_last = train_transform, True, True
    else:
        transform, shuffle, drop_last = test_transform, False, False

    num_datasets = len(data_infos)

    # Combine data from multiple datasets
    data_info = []
    for i in range(num_datasets):
        image_dir_ = args.image_dir[i]
        data_info_ = data_infos[i]
        dataset_name = dataset_names[i]
        data_info_ = fit_multi_data_info_format(args, data_info_, dataset_name, image_dir_)
        if not args.multi_view:
            data_info_ = multi_view_to_single_view_data_info(data_info_)
        data_info.append(data_info_)

    # Concatenate all data into a single DataFrame
    data_info = pd.concat(data_info, ignore_index=True)

    # Create the dataset and DataLoader
    dataset = myDataset(args, data_info, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle,
                             num_workers=args.num_workers, pin_memory=True, drop_last=drop_last)

    return data_loader


def datainfo_to_custom(template_datainfo, custom_data_info, file_path, dataset='Inhouse'):
    def give_label(csv_data_info):
        # race
        csv_data_info['race_raw'] = csv_data_info['race']
        csv_data_info['race'] = csv_data_info['race_raw'].map({'unknow': -1, 'white': 0, 'asian': 1, 'black': 2, 'other': 3})
        # density
        csv_data_info['density_raw'] = csv_data_info['density']
        csv_data_info['density'][csv_data_info['density_raw'].isna()] = -1
        csv_data_info['density'] = csv_data_info['density_raw'].map({-1: -1, 1: 0, 2: 1, 3: 2, 4: 3})
        # birads
        csv_data_info['birads_raw'] = csv_data_info['birads']
        # malignancy
        csv_data_info['malignancy_raw'] = csv_data_info['malignancy']
        csv_data_info['malignancy_r_raw'] = csv_data_info['malignancy_r']
        csv_data_info['malignancy_l_raw'] = csv_data_info['malignancy_l']
        # years_to_cancer
        csv_data_info['years_to_cancer_raw'] = csv_data_info['years_to_cancer']
        csv_data_info['years_to_cancer_r_raw'] = csv_data_info['years_to_cancer_r']
        csv_data_info['years_to_cancer_l_raw'] = csv_data_info['years_to_cancer_l']
        # years_to_last_followup
        csv_data_info['years_to_last_followup_raw'] = csv_data_info['years_to_last_followup']
        return csv_data_info

    custom_data_info = custom_data_info.dropna(subset=['PATH_L_CC', 'PATH_R_CC', 'PATH_L_MLO', 'PATH_R_MLO'])
    custom_data_info = give_label(custom_data_info)

    # custom_data_info['PATH_L_CC'] = custom_data_info['PATH_L_CC'].apply(lambda x: os.path.join(file_path, x)) # os.path.join(file_path, x) not working don't know why
    custom_data_info['PATH_L_CC'] = custom_data_info['PATH_L_CC'].apply(lambda x: f"{file_path}/{x}")
    custom_data_info['PATH_R_CC'] = custom_data_info['PATH_R_CC'].apply(lambda x: f"{file_path}/{x}")
    custom_data_info['PATH_L_MLO'] = custom_data_info['PATH_L_MLO'].apply(lambda x: f"{file_path}/{x}")
    custom_data_info['PATH_R_MLO'] = custom_data_info['PATH_R_MLO'].apply(lambda x: f"{file_path}/{x}")

    template_datainfo = custom_data_info[[
        'patient_id', 'exam_id', 'age', 'density', 'race', 'birads', 'malignancy', 'malignancy_r','malignancy_l',
        'years_to_cancer', 'years_to_cancer_r', 'years_to_cancer_l', 'years_to_last_followup',
        'PATH_L_CC', 'PATH_R_CC', 'PATH_L_MLO', 'PATH_R_MLO']]

    template_datainfo['dataset'] = dataset
    return template_datainfo


def multi_view_to_single_view_data_info(custom_data_info):
    """
    Organisize custom dataset CSV into some format:
        Example --->
                --->patient_id: 0001 # patient_id
                --->exam_id: 0001 # exam_id
                --->age: 85.5 # age
                --->density: 4 # [-1: density unknown, 0: almost entirely fat, 1: scattered fibroglandular densities, 2: heterogeneously dense, 3: extremely dense]
                --->race: 1 # [0: white, 1: asian, 2:black, 3:other]
                --->malignancy: 1 # [0: benign, 1: malignant]
                --->birads: 1 # [-1: birads unknown, 0: birads0, 1: birads1, 2: birads2, 3: birads3, 4: birads4, 5: birads5, 6: birads6]
                --->years_to_cancer: 1 # 0-100
                --->years_to_last_followup: 2 # 0-100
                --->PATH: 'xxx/xxx/xxx.png' # image path
                --->dataset: 'Inhosue'  # ['Inhouse', 'VinDr', 'RSNA']
    """

    metadata_info = {
        'patient_id': [],
        'exam_id': [],
        'laterality': [],
        'view': [],
        'PATH': [],
        'density': [],
        'age': [],
        'race': [],
        'malignancy': [],
        'birads': [],
        'years_to_cancer': [],
        'years_to_last_followup': [],
        'dataset': [],
    }

    for i in range(len(custom_data_info)):
        patient_id = custom_data_info['patient_id'].iloc[i]
        exam_id = custom_data_info['exam_id'].iloc[i]
        age = custom_data_info['age'].iloc[i]
        density = custom_data_info['density'].iloc[i]
        race = custom_data_info['race'].iloc[i]
        dataset = custom_data_info['dataset'].iloc[i]
        birads = custom_data_info['birads'].iloc[i]
        years_to_last_followup = custom_data_info['years_to_last_followup'].iloc[i]

        lateralitys = ['L', 'R']
        views = ['CC', 'MLO']
        for laterality_idx in range(len(lateralitys)):
            laterality = lateralitys[laterality_idx]
            for view_idx in range(len(views)):
                if laterality == 'R':
                    years_to_cancer = custom_data_info['years_to_cancer_r'].iloc[i]
                    malignancy = custom_data_info['malignancy_r'].iloc[i]
                else:
                    years_to_cancer = custom_data_info['years_to_cancer_l'].iloc[i]
                    malignancy = custom_data_info['malignancy_l'].iloc[i]

                view = views[view_idx]
                PATH = custom_data_info['PATH_{}_{}'.format(laterality, view)].iloc[i]

                metadata_info['patient_id'].append(patient_id)
                metadata_info['exam_id'].append(exam_id)
                metadata_info['view'].append(view)
                metadata_info['laterality'].append(laterality)
                metadata_info['age'].append(age)
                metadata_info['density'].append(density)
                metadata_info['race'].append(race)
                metadata_info['dataset'].append(dataset)
                metadata_info['PATH'].append(PATH)
                metadata_info['malignancy'].append(malignancy)
                metadata_info['birads'].append(birads)
                metadata_info['years_to_cancer'].append(years_to_cancer)
                metadata_info['years_to_last_followup'].append(years_to_last_followup)

    metadata_info = pd.DataFrame(data=metadata_info)
    return metadata_info


def fit_multi_data_info_format(args, custom_data_info, dataset, file_path):
    """
    Organisize custom dataset CSV into some format:
    Example --->
            --->patient_id: 0001 # patient_id
            --->exam_id: 0001 # exam_id
            --->age: 85.5 # age
            --->density: 4 # [-1: density unknown, 0: almost entirely fat, 1: scattered fibroglandular densities, 2: heterogeneously dense, 3: extremely dense]
            --->race: 1 # [0: white, 1: asian, 2:black, 3:other]
            --->malignancy: 1 # [0: benign, 1: malignant]
            --->birads: 1 # [-1: birads unknown, 0: birads0, 1: birads1, 2: birads2, 3: birads3, 4: birads4, 5: birads5, 6: birads6]
            --->years_to_cancer: 1 # 0-100
            --->years_to_last_followup: 2 # 0-100
            --->PATH_L_CC: 'xxx/xxx/xxx.png' # image path
            --->PATH_R_CC: 'xxx/xxx/xxx.png' # image path
            --->PATH_L_MLO: 'xxx/xxx/xxx.png' # image path
            --->PATH_R_MLO: 'xxx/xxx/xxx.png' # image path
            --->dataset: 'Inhosue'  # ['Inhouse', 'VinDr', 'RSNA']
    """
    template_datainfo = {
        'patient_id':[],
        'exam_id':[],
        'age':[],
        'density':[],
        'race':[],
        'malignancy':[],
        'malignancy_r':[],
        'malignancy_l':[],
        'birads':[],
        'years_to_cancer':[],
        'years_to_cancer_r':[],
        'years_to_cancer_l':[],
        'years_to_last_followup':[],
        'PATH_L_CC':[],
        'PATH_R_CC':[],
        'PATH_L_MLO':[],
        'PATH_R_MLO':[],
        'dataset':[],
    }
    datasets = ["inhouse", "embed", "csaw", "cmmd", "vindr", "rsna"] # TODO: Add more datasets
    if dataset in datasets:
        data_info = datainfo_to_custom(template_datainfo, custom_data_info, file_path, dataset=dataset)
    else:
        raise ValueError(f" DATASET: {args.dataset} is not supported.")

    return data_info


