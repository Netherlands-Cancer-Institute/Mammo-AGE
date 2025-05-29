import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    """
    Custom dataset for loading mammogram images with associated metadata.

    Args:
        args: Arguments containing image size and other parameters.
        data_info: DataFrame containing image paths and metadata.
        transform: Transformations to be applied to the images.
    """

    def __init__(self, args, data_info, transform):
        self.img_size = (args.img_size, args.img_size // 2)
        self.img_size_cv = (args.img_size // 2, args.img_size)
        # age = data_info['image_age']
        age = data_info['age']

        self.age = np.asarray(age, dtype='float32')
        self.data_info = data_info  # header=None是去掉表头部分
        image_file_path = np.vstack(
            (np.asarray(data_info.PATH_L_CC),
             np.asarray(data_info.PATH_R_CC),
             np.asarray(data_info.PATH_L_MLO),
             np.asarray(data_info.PATH_R_MLO),
             ))
        self.img_file_path = image_file_path

        self.density = np.asarray(data_info['density'], dtype='int64')
        self.race = np.asarray(data_info['race'], dtype='int64')
        # self.patient_id = data_info['patient_id']
        # self.exam_id = data_info['exam_id']

        self.transform = transform

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at the specified index.

        Args:
            index: Index of the item to retrieve.

        Returns:
            A dictionary containing the image tensor and associated metadata.
        """
        path = self.img_file_path[:, index]

        age = self.age[index]
        density = self.density[index]
        race = self.race[index]
        pid = self.data_info['patient_id'].iloc[index]
        # pid = self.patient_id[index]
        exam_id = self.data_info['exam_id'].iloc[index]
        # exam_id = self.exam_id[index]

        age = np.asarray(age, dtype='float32')
        density = np.asarray(density, dtype='int64')
        race = np.asarray(race, dtype='int64')

        img_l_cc, img_l_cc_path = self.__getimg(path[0])
        img_r_cc, img_r_cc_path = self.__getimg(path[1])
        img_l_mlo, img_l_mlo_path = self.__getimg(path[2])
        img_r_mlo, img_r_mlo_path = self.__getimg(path[3])

        img_l_cc = Image.fromarray(img_l_cc)
        img_r_cc = Image.fromarray(img_r_cc)
        img_l_mlo = Image.fromarray(img_l_mlo)
        img_r_mlo = Image.fromarray(img_r_mlo)

        if self.transform is not None:
            # img_l_cc = self.transform(img_l_cc)
            # img_r_cc = self.transform(img_r_cc)
            # img_l_mlo = self.transform(img_l_mlo)
            # img_r_mlo = self.transform(img_r_mlo)
            img_l_cc = self.transform(image=np.array(img_l_cc))["image"]
            img_r_cc = self.transform(image=np.array(img_r_cc))["image"]
            img_l_mlo = self.transform(image=np.array(img_l_mlo))["image"]
            img_r_mlo = self.transform(image=np.array(img_r_mlo))["image"]

        img = torch.cat((
            img_l_cc.unsqueeze(0), img_r_cc.unsqueeze(0), img_l_mlo.unsqueeze(0), img_r_mlo.unsqueeze(0)), 0)

        pid = str(pid)
        exam_id = str(exam_id)
        # return img, age, density, race, img_l_cc_path, img_r_cc_path, img_l_mlo_path, img_r_mlo_path, pid, exam_id

        return {'img': img,
                'age': age,
                'density': density,
                'race': race,
                'img_l_cc_path': img_l_cc_path,
                'img_r_cc_path': img_r_cc_path,
                'img_l_mlo_path': img_l_mlo_path,
                'img_r_mlo_path': img_r_mlo_path,
                'pid': pid,
                'exam_id': exam_id,
                }

    def __len__(self):
        return len(self.data_info)


    def __getimg(self, path):
        img_path = path
        try:
            image = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            if image is None:
                raise ValueError(f"Image not found or unable to read: {img_path}")
            image = cv2.resize(image.astype(np.uint16), self.img_size_cv)
            image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
            img = image.astype(np.uint8)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = np.zeros(self.img_size_cv, dtype=np.uint8)  # Return a blank image in case of error
        return img, img_path