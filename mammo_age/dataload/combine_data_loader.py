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


def NKI_datainfo_to_custom(template_datainfo, custom_data_info, file_path):
    def give_label(csv_data_info):
        # race
        csv_data_info['race'][csv_data_info['race'] == 'unknow'] = -1
        csv_data_info['race'][csv_data_info['race'] == 'white'] = 0
        csv_data_info['race'][csv_data_info['race'] == 'asian'] = 1
        csv_data_info['race'][csv_data_info['race'] == 'black'] = 2
        csv_data_info['race'][csv_data_info['race'] == 'other'] = 3
        # density
        csv_data_info['density'][csv_data_info['density'].isna()] = -1
        csv_data_info['density'][csv_data_info['density'] == 1] = 0
        csv_data_info['density'][csv_data_info['density'] == 2] = 1
        csv_data_info['density'][csv_data_info['density'] == 3] = 2
        csv_data_info['density'][csv_data_info['density'] == 4] = 3
        return csv_data_info

    def get_img_path(file_path, path):
        img_path = str(path).replace(
            '/home/x.wang/new_nki_project/NKI_data_archive/img_processing/processed_20220322_resize1024_512/imgs/',
            file_path)
        return img_path

    custom_data_info = give_label(custom_data_info)

    for i in range(len(custom_data_info)):
        patient_id = custom_data_info['patient_id'].iloc[i]
        exam_id = custom_data_info['exam_id'].iloc[i]
        age = custom_data_info['age'].iloc[i]
        density = custom_data_info['density'].iloc[i]
        race = custom_data_info['race'].iloc[i]
        PATH_R_CC = get_img_path(file_path, custom_data_info['PATH_R_CC_processed'].iloc[i])
        PATH_L_CC = get_img_path(file_path, custom_data_info['PATH_L_CC_processed'].iloc[i])
        PATH_R_MLO = get_img_path(file_path, custom_data_info['PATH_R_MLO_processed'].iloc[i])
        PATH_L_MLO = get_img_path(file_path, custom_data_info['PATH_L_MLO_processed'].iloc[i])
        dataset = 'NKI'

        template_datainfo['patient_id'].append(patient_id)
        template_datainfo['exam_id'].append(exam_id)
        template_datainfo['age'].append(age)
        template_datainfo['density'].append(density)
        template_datainfo['race'].append(race)
        template_datainfo['PATH_R_CC'].append(PATH_R_CC)
        template_datainfo['PATH_L_CC'].append(PATH_L_CC)
        template_datainfo['PATH_R_MLO'].append(PATH_R_MLO)
        template_datainfo['PATH_L_MLO'].append(PATH_L_MLO)
        template_datainfo['dataset'].append(dataset)

    data_info = pd.DataFrame(data=template_datainfo)
    return data_info


def Inhouse_datainfo_to_custom(template_datainfo, custom_data_info, file_path):
    def give_label(csv_data_info):
        # race
        csv_data_info['race'][csv_data_info['race'] == 'unknow'] = -1
        csv_data_info['race'][csv_data_info['race'] == 'white'] = 0
        csv_data_info['race'][csv_data_info['race'] == 'asian'] = 1
        csv_data_info['race'][csv_data_info['race'] == 'black'] = 2
        csv_data_info['race'][csv_data_info['race'] == 'other'] = 3
        # density
        csv_data_info['density'][csv_data_info['density'].isna()] = -1
        csv_data_info['density'][csv_data_info['density'] == 1] = 0
        csv_data_info['density'][csv_data_info['density'] == 2] = 1
        csv_data_info['density'][csv_data_info['density'] == 3] = 2
        csv_data_info['density'][csv_data_info['density'] == 4] = 3
        return csv_data_info

    def get_img_path(file_path, path):
        img_path = str(path).replace(
            '/data/groups/aiforoncology/archive/radiology/nki-breast/Mammo/inhouse/MG/preprocessed/img-2048/',
            file_path)
        return img_path

    custom_data_info = custom_data_info.dropna(subset=['PATH_L_CC_processed', 'PATH_R_CC_processed',
                                                       'PATH_L_MLO_processed', 'PATH_R_MLO_processed'])

    custom_data_info = give_label(custom_data_info)

    for i in range(len(custom_data_info)):
        patient_id = custom_data_info['patient_id'].iloc[i]
        exam_id = custom_data_info['exam_id'].iloc[i]
        age = custom_data_info['age'].iloc[i]
        density = custom_data_info['density'].iloc[i]
        race = custom_data_info['race'].iloc[i]
        PATH_R_CC = get_img_path(file_path, custom_data_info['PATH_R_CC_processed'].iloc[i])
        PATH_L_CC = get_img_path(file_path, custom_data_info['PATH_L_CC_processed'].iloc[i])
        PATH_R_MLO = get_img_path(file_path, custom_data_info['PATH_R_MLO_processed'].iloc[i])
        PATH_L_MLO = get_img_path(file_path, custom_data_info['PATH_L_MLO_processed'].iloc[i])
        dataset = 'NKI'

        template_datainfo['patient_id'].append(patient_id)
        template_datainfo['exam_id'].append(exam_id)
        template_datainfo['age'].append(age)
        template_datainfo['density'].append(density)
        template_datainfo['race'].append(race)
        template_datainfo['PATH_R_CC'].append(PATH_R_CC)
        template_datainfo['PATH_L_CC'].append(PATH_L_CC)
        template_datainfo['PATH_R_MLO'].append(PATH_R_MLO)
        template_datainfo['PATH_L_MLO'].append(PATH_L_MLO)
        template_datainfo['dataset'].append(dataset)

    data_info = pd.DataFrame(data=template_datainfo)
    return data_info


def RSNA_datainfo_to_custom(template_datainfo, custom_data_info, file_path):
    def give_label(csv_data_info):
        # density
        csv_data_info['density'][csv_data_info['density'].isna()] = -1
        csv_data_info['density'][csv_data_info['density'] == 'A'] = 0
        csv_data_info['density'][csv_data_info['density'] == 'B'] = 1
        csv_data_info['density'][csv_data_info['density'] == 'C'] = 2
        csv_data_info['density'][csv_data_info['density'] == 'D'] = 3
        return csv_data_info

    def get_patient_list(data_info):
        all_patient_id = list(data_info["patient_id"])
        patient_id = []
        [patient_id.append(i) for i in all_patient_id if not i in patient_id]
        return patient_id

    custom_data_info = give_label(custom_data_info)
    all_patient_id = get_patient_list(custom_data_info)
    dataset = 'RSNA'

    for i in range(len(all_patient_id)):
        patient_id = all_patient_id[i]
        template_datainfo['patient_id'].append(patient_id)
        template_datainfo['exam_id'].append(patient_id)
        template_datainfo['dataset'].append(dataset)
        template_datainfo['race'].append(-1)   # RSNA do not have race label

        data_info_ = custom_data_info[custom_data_info['patient_id'] == patient_id]

        lateralitys = ['L', 'R']
        views = ['CC', 'MLO']
        for laterality_idx in range(len(lateralitys)):
            laterality = lateralitys[laterality_idx]
            for view_idx in range(len(views)):
                view = views[view_idx]
                data_info_img = data_info_[(data_info_['view'] == view)
                                           & (data_info_['laterality'] == laterality)]

                try:
                    image_id = data_info_img['image_id'].iloc[0]
                except:
                    print("An exception occurred")
                    print("len datainfo", len(data_info_img))
                    print(data_info_img)
                    print('patinet_id:{}, view_position:{}, laterality:{}'.format(patient_id, view, laterality))

                img_path = '{}/{}/{}.png'.format(file_path, patient_id, image_id)

                if laterality == 'L' and view == 'CC':
                    age = data_info_img['age'].iloc[0]
                    template_datainfo['age'].append(age)
                    density = data_info_img['density'].iloc[0]
                    template_datainfo['density'].append(density)

                template_datainfo['PATH_{}_{}'.format(laterality, view)].append(img_path)

    data_info = pd.DataFrame(data=template_datainfo)
    return data_info


def VinDr_datainfo_to_custom(template_datainfo, custom_data_info, file_path):
    def give_label(csv_data_info):
        # density
        csv_data_info['breast_density'][csv_data_info['breast_density'].isna()] = -1
        csv_data_info['breast_density'][csv_data_info['breast_density'] == 'DENSITY A'] = 0
        csv_data_info['breast_density'][csv_data_info['breast_density'] == 'DENSITY B'] = 1
        csv_data_info['breast_density'][csv_data_info['breast_density'] == 'DENSITY C'] = 2
        csv_data_info['breast_density'][csv_data_info['breast_density'] == 'DENSITY D'] = 3
        return csv_data_info

    def get_patient_list(data_info):
        all_patient_id = list(data_info["study_id"])
        patient_id = []
        [patient_id.append(i) for i in all_patient_id if not i in patient_id]
        return patient_id

    custom_data_info = give_label(custom_data_info)
    all_patient_id = get_patient_list(custom_data_info)
    dataset = 'VinDr'

    for i in range(len(all_patient_id)):
        patient_id = all_patient_id[i]
        if patient_id == 'dc8eca8e41cb72804147da33e91917bb':
            continue
        if patient_id == '02bd0bd83c6d9fedc49b0df6ecd952c6':
            continue
        template_datainfo['patient_id'].append(patient_id)
        template_datainfo['exam_id'].append(patient_id)
        template_datainfo['dataset'].append(dataset)
        template_datainfo['race'].append(-1)  # VinDr do not have race label

        data_info_ = custom_data_info[custom_data_info['study_id'] == patient_id]

        lateralitys = ['L', 'R']
        views = ['CC', 'MLO']
        for laterality_idx in range(len(lateralitys)):
            laterality = lateralitys[laterality_idx]
            for view_idx in range(len(views)):
                view = views[view_idx]
                data_info_img = data_info_[(data_info_['view_position'] == view)
                                           & (data_info_['laterality'] == laterality)]

                try:
                    image_id = data_info_img['image_id'].iloc[0]
                except:
                    print("An exception occurred")
                    print("len datainfo", len(data_info_img))
                    print(data_info_img)
                    print('patinet_id:{}, view_position:{}, laterality:{}'.format(patient_id, view, laterality))

                img_path = '{}/{}/{}.png'.format(file_path, patient_id, image_id)

                if laterality == 'L' and view == 'CC':
                    age = str(data_info_img['age'].iloc[0])[:-1]
                    age = np.asarray(age, dtype='float32')
                    template_datainfo['age'].append(age)
                    density = data_info_img['breast_density'].iloc[0]
                    template_datainfo['density'].append(density)

                template_datainfo['PATH_{}_{}'.format(laterality, view)].append(img_path)

    data_info = pd.DataFrame(data=template_datainfo)
    return data_info


def DDSM_datainfo_to_custom(template_datainfo, custom_data_info, file_path):
    def give_label(csv_data_info):
        # density
        csv_data_info['Density'][csv_data_info['Density'].isna()] = -1
        csv_data_info['Density'][csv_data_info['Density']==0] = -1
        csv_data_info['Density'][csv_data_info['Density']==1] = 0
        csv_data_info['Density'][csv_data_info['Density']==2] = 1
        csv_data_info['Density'][csv_data_info['Density']==3] = 2
        csv_data_info['Density'][csv_data_info['Density']==4] = 3
        return csv_data_info

    def get_patient_list(data_info):
        all_patient_id = list(data_info["patient_id"])
        patient_id = []
        [patient_id.append(i) for i in all_patient_id if not i in patient_id]
        if 'A_1825_1' in patient_id:
            patient_id.remove('A_1825_1')
        if 'B_3102_1' in patient_id:
            patient_id.remove('B_3102_1')

        return patient_id

    custom_data_info = give_label(custom_data_info)
    all_patient_id = get_patient_list(custom_data_info)
    dataset = 'DDSM'

    for i in range(len(all_patient_id)):
        patient_id = all_patient_id[i]
        template_datainfo['patient_id'].append(patient_id)
        template_datainfo['exam_id'].append(patient_id)
        template_datainfo['dataset'].append(dataset)
        template_datainfo['race'].append(-1)   # RSNA do not have race label

        data_info_ = custom_data_info[custom_data_info['patient_id'] == patient_id]

        lateralitys = ['LEFT', 'RIGHT']
        views = ['CC', 'MLO']
        for laterality_idx in range(len(lateralitys)):
            laterality = lateralitys[laterality_idx]
            for view_idx in range(len(views)):
                view = views[view_idx]
                data_info_img = data_info_[(data_info_['View'] == view)
                                           & (data_info_['Side'] == laterality)]

                try:
                    image_id = data_info_img['subid'].iloc[0]
                    subfolder = data_info_img['subfolder'].iloc[0]
                except:
                    print("An exception occurred")
                    print("len datainfo", len(data_info_img))
                    print(data_info_img)
                    print('patinet_id:{}, view_position:{}, laterality:{}'.format(patient_id, view, laterality))

                img_path = '{}/imgs/{}_{}.png'.format(file_path, subfolder, image_id)

                if laterality == 'LEFT' and view == 'CC':
                    age = data_info_img['Age'].iloc[0]
                    template_datainfo['age'].append(age)
                    density = data_info_img['Density'].iloc[0]
                    template_datainfo['density'].append(density)

                template_datainfo['PATH_{}_{}'.format(laterality[0], view)].append(img_path)

    data_info = pd.DataFrame(data=template_datainfo)
    return data_info


def CMMD_datainfo_to_custom(template_datainfo, custom_data_info, file_path):
    def get_patient_list(data_info):
        all_patient_id = list(data_info["ID1"])
        patient_id = []
        [patient_id.append(i) for i in all_patient_id if not i in patient_id]
        return patient_id

    all_patient_id = get_patient_list(custom_data_info)
    dataset = 'CMMD'

    for i in range(len(all_patient_id)):
        patient_id = all_patient_id[i]
        template_datainfo['patient_id'].append(patient_id)
        template_datainfo['exam_id'].append(patient_id)
        template_datainfo['dataset'].append(dataset)
        template_datainfo['race'].append(-1)   # RSNA do not have race label

        data_info_ = custom_data_info[custom_data_info['ID1'] == patient_id]
        lateralitys = ['L', 'R']
        views = ['CC', 'MLO']
        age = data_info_['Age'].iloc[0]
        template_datainfo['age'].append(age)
        template_datainfo['density'].append(-1)

        if len(data_info_) == 1:
            lateralitys_prov = [data_info_['LeftRight'].iloc[0]]
        elif len(data_info_) == 2:
            lateralitys_prov = ['L', 'R']

        for laterality_idx in range(len(lateralitys)):
            laterality = lateralitys[laterality_idx]
            for view_idx in range(len(views)):
                view = views[view_idx]

                if laterality in lateralitys_prov:
                    image_id = f'{patient_id}_{view}_{laterality}'
                else:
                    image_id = f'{patient_id}_{view}_{lateralitys_prov[0]}'

                img_path = '{}/{}.png'.format(file_path, image_id)

                template_datainfo['PATH_{}_{}'.format(laterality, view)].append(img_path)

    data_info = pd.DataFrame(data=template_datainfo)
    return data_info


def BMCD_datainfo_to_custom(template_datainfo, custom_data_info, file_path):
    def give_label(csv_data_info):
        # density
        csv_data_info['density'][csv_data_info['density'].isna()] = -1
        csv_data_info['density'][csv_data_info['density'] == 'a'] = 0
        csv_data_info['density'][csv_data_info['density'] == 'b'] = 1
        csv_data_info['density'][csv_data_info['density'] == 'c'] = 2
        csv_data_info['density'][csv_data_info['density'] == 'd'] = 3
        return csv_data_info

    custom_data_info = give_label(custom_data_info)
    dataset = 'BMCD'

    for i in range(len(custom_data_info)):

        folder = custom_data_info['folder'].iloc[i]
        sub_name_img = custom_data_info['sub_name_img'].iloc[i]
        subfolder = custom_data_info['subfolder'].iloc[i]
        lateralitys_prov = [custom_data_info['laterality'].iloc[i]]
        age = custom_data_info['Age'].iloc[i]
        density = custom_data_info['density'].iloc[i]

        template_datainfo['patient_id'].append(f'{folder}_{subfolder}')
        template_datainfo['exam_id'].append(sub_name_img)
        template_datainfo['dataset'].append(dataset)
        template_datainfo['race'].append(-1)  # RSNA do not have race label
        template_datainfo['age'].append(age)
        template_datainfo['density'].append(density)

        lateralitys = ['L', 'R']
        views = ['CC', 'MLO']

        for laterality_idx in range(len(lateralitys)):
            laterality = lateralitys[laterality_idx]
            for view_idx in range(len(views)):
                view = views[view_idx]
                image_id = f'{folder}_{subfolder}_{view}_{sub_name_img}'
                img_path = '{}/{}.png'.format(file_path, image_id)
                template_datainfo['PATH_{}_{}'.format(laterality, view)].append(img_path)

    data_info = pd.DataFrame(data=template_datainfo)
    return data_info


def CESM_datainfo_to_custom(template_datainfo, custom_data_info, file_path):
    def give_label(csv_data_info):
        # density
        csv_data_info['Breast density (ACR)'][
            csv_data_info['Breast density (ACR)'].isna()] = -1
        csv_data_info['Breast density (ACR)'][
            csv_data_info['Breast density (ACR)'] == '_'] = -1
        csv_data_info['Breast density (ACR)'][
            csv_data_info['Breast density (ACR)'] == 'A'] = 0
        csv_data_info['Breast density (ACR)'][
            csv_data_info['Breast density (ACR)'] == 'B'] = 1
        csv_data_info['Breast density (ACR)'][
            csv_data_info['Breast density (ACR)'] == 'C'] = 2
        csv_data_info['Breast density (ACR)'][
            csv_data_info['Breast density (ACR)'] == 'D'] = 3
        return csv_data_info

    def get_patient_list(data_info):
        all_patient_id = list(data_info["Patient_ID"])
        patient_id = []
        [patient_id.append(i) for i in all_patient_id if not i in patient_id]
        return patient_id

    custom_data_info = give_label(custom_data_info)
    all_patient_id = get_patient_list(custom_data_info)
    dataset = 'CDD-CESM'

    for i in range(len(all_patient_id)):
        patient_id = all_patient_id[i]
        template_datainfo['patient_id'].append(patient_id)
        template_datainfo['exam_id'].append(patient_id)
        template_datainfo['dataset'].append(dataset)
        template_datainfo['race'].append(-1)  # RSNA do not have race label

        data_info_ = custom_data_info[custom_data_info['Patient_ID'] == patient_id]

        imgs_paths = {
            'PATH_L_CC':[],
            'PATH_R_CC':[],
            'PATH_L_MLO':[],
            'PATH_R_MLO':[],
        }

        densities = []
        ages = []

        lateralitys = ['L', 'R']
        views = ['CC', 'MLO']
        for laterality_idx in range(len(lateralitys)):
            laterality = lateralitys[laterality_idx]

            for view_idx in range(len(views)):
                view = views[view_idx]
                data_info_img = data_info_[(data_info_['View'] == view)
                                           & (data_info_['Side'] == laterality)]

                if len(data_info_img) == 1:
                    img_id = data_info_img['Image_name'].iloc[0]

                    img_id = str(img_id).replace(' ', '')
                    img_path = '{}{}.png'.format(file_path, img_id)
                    imgs_paths['PATH_{}_{}'.format(laterality, view)] = img_path

                    densities.append(data_info_img['Breast density (ACR)'].iloc[0])
                    ages.append(data_info_img['Age'].iloc[0])

        density = max(densities)
        age = max(ages)

        template_datainfo['age'].append(age)
        template_datainfo['density'].append(density)

        lateralitys = ['L', 'R']
        lateralitys_con  = ['R', 'L']
        views = ['CC', 'MLO']
        views_con = ['MLO', 'CC']
        for laterality_idx in range(len(lateralitys)):
            laterality = lateralitys[laterality_idx]
            laterality_con = lateralitys_con[laterality_idx]

            for view_idx in range(len(views)):
                view = views[view_idx]
                view_con = views_con[view_idx]

                if len(imgs_paths['PATH_{}_{}'.format(laterality, view)]) != 0:
                    path_ = imgs_paths['PATH_{}_{}'.format(laterality, view)]
                elif len(imgs_paths['PATH_{}_{}'.format(laterality_con, view)]) != 0:
                    path_ = imgs_paths['PATH_{}_{}'.format(laterality_con, view)]
                elif len(imgs_paths['PATH_{}_{}'.format(laterality, view_con)]) != 0:
                    path_ = imgs_paths['PATH_{}_{}'.format(laterality, view_con)]
                else:
                    path_ = imgs_paths['PATH_{}_{}'.format(laterality_con, view_con)]

                template_datainfo['PATH_{}_{}'.format(laterality, view)].append(path_)

    data_info = pd.DataFrame(data=template_datainfo)
    return data_info


def EMBED_datainfo_to_custom(template_datainfo, custom_data_info, file_path):
    dataset = 'EMBED'
    def give_label(csv_data_info):
        # density
        csv_data_info['density'][csv_data_info['density'].isna()] = -1
        csv_data_info['density'][csv_data_info['density']==5] = -1
        csv_data_info['density'][csv_data_info['density']==1] = 0
        csv_data_info['density'][csv_data_info['density']==2] = 1
        csv_data_info['density'][csv_data_info['density']==3] = 2
        csv_data_info['density'][csv_data_info['density']==4] = 3
        return csv_data_info

    custom_data_info = give_label(custom_data_info)
    all_patient_id = custom_data_info.patient_id.drop_duplicates().tolist()

    for i in range(len(all_patient_id)):
        patient_id = all_patient_id[i]
        data_info_p = custom_data_info[custom_data_info['patient_id'] == patient_id]
        for e in range(len(data_info_p)):
            data_info_e = data_info_p.iloc[e]
            template_datainfo['patient_id'].append(patient_id)
            exam_id = data_info_e['exam_id']
            template_datainfo['exam_id'].append(data_info_e['exam_id'])
            template_datainfo['dataset'].append(dataset)
            template_datainfo['race'].append(-1)
            template_datainfo['age'].append(data_info_e['age'])
            template_datainfo['density'].append(data_info_e['density'])

            lateralitys = ['L', 'R']
            views = ['CC', 'MLO']
            for laterality_idx in range(len(lateralitys)):
                laterality = lateralitys[laterality_idx]
                for view_idx in range(len(views)):
                    view = views[view_idx]
                    img_path = f'{file_path}/{patient_id}/{exam_id}_{laterality}_{view}.png'
                    # img_path = '{}{}'.format(file_path, data_info_e['PATH_{}_{}'.format(laterality, view)])
                    template_datainfo['PATH_{}_{}'.format(laterality, view)].append(img_path)

    data_info = pd.DataFrame(data=template_datainfo)
    return data_info


def multi_view_to_single_view_data_info(custom_data_info):
    """
    Organisize custom dataset CSV into some format:
        Example --->
                --->patient_id: 0001
                --->exam_id: 0001
                --->age: 85.5
                --->density: 4 # [1,2,3,4]
                --->race: 1 # [0: white, 1: asian, 2:black, 3:other]
                --->path: 'xxx/xxx/xxx.png'
                --->dataset: 'NKI'  # ['NKI', 'RSNA', 'VinDr']
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
        'dataset': []
    }

    for i in range(len(custom_data_info)):
        patient_id = custom_data_info['patient_id'].iloc[i]
        exam_id = custom_data_info['exam_id'].iloc[i]
        age = custom_data_info['age'].iloc[i]
        density = custom_data_info['density'].iloc[i]
        race = custom_data_info['race'].iloc[i]
        dataset = custom_data_info['dataset'].iloc[i]

        lateralitys = ['L', 'R']
        views = ['CC', 'MLO']
        for laterality_idx in range(len(lateralitys)):
            laterality = lateralitys[laterality_idx]
            for view_idx in range(len(views)):
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

    metadata_info = pd.DataFrame(data=metadata_info)
    return metadata_info


def fit_multi_data_info_format(args, custom_data_info, dataset, file_path):
    """
    Organisize custom dataset CSV into some format:
    Example --->
            --->patient_id: 0001
            --->exam_id: 0001
            --->age: 85.5
            --->density: 4 # [1,2,3,4]
            --->race: 1 # [0: white, 1: asian, 2:black, 3:other]
            --->path_r_cc: 'xxx/xxx/xxx.png'
            --->path_r_mlo: 'xxx/xxx/xxx.png'
            --->path_l_cc: 'xxx/xxx/xxx.png'
            --->path_l_mlo: 'xxx/xxx/xxx.png'
            --->dataset: 'NKI'  # ['NKI', 'RSNA', 'VinDr']
    """
    template_datainfo = {
        'patient_id':[],
        'exam_id':[],
        'age':[],
        'density':[],
        'race':[],
        'PATH_L_CC':[],
        'PATH_R_CC':[],
        'PATH_L_MLO':[],
        'PATH_R_MLO':[],
        'dataset':[],
    }
    datasets = ['NKI', 'RSNA', 'VinDr', 'DDSM', 'CMMD', 'BMCD', 'CESM', 'EMBED', "Inhouse"]
    if dataset in datasets:
        data_info = eval('{}_datainfo_to_custom'.format(dataset))(template_datainfo, custom_data_info, file_path)
    else:
        raise ValueError(f" DATASET: {args.dataset} is not supported.")

    return data_info
