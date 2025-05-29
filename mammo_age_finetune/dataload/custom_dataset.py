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

        self.malignancy = np.asarray(data_info['malignancy'], dtype='int64')
        self.birads = np.asarray(data_info['birads'], dtype='int64')
        self.years_to_cancer = np.asarray(data_info['years_to_cancer'], dtype='int64')
        self.years_to_last_followup = np.asarray(data_info['years_to_last_followup'], dtype='int64')

        self.age = np.asarray(age, dtype='float32')
        self.data_info = data_info
        image_file_path = np.asarray(data_info.PATH)
        self.img_file_path = image_file_path
        self.density = np.asarray(data_info['density'], dtype='int64')
        self.race = np.asarray(data_info['race'], dtype='int64')
        self.transform = transform

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at the specified index.

        Args:
            index: Index of the item to retrieve.

        Returns:
            A dictionary containing the image tensor and associated metadata.
        """
        path = self.img_file_path[index]

        age = self.age[index]
        density = self.density[index]
        race = self.race[index]

        pid = self.data_info['patient_id'].iloc[index]
        exam_id = self.data_info['exam_id'].iloc[index]

        # Label for the image
        age = np.asarray(age, dtype='float32')
        density = np.asarray(density, dtype='int64')
        race = np.asarray(race, dtype='int64')
        malignancy = np.asarray(self.malignancy[index], dtype='int64')
        birads = np.asarray(self.birads[index], dtype='int64')
        years_to_cancer = np.asarray(self.years_to_cancer[index], dtype='int64')
        years_to_last_followup = np.asarray(self.years_to_last_followup[index], dtype='int64')
        view = self.data_info['view'].iloc[index]
        laterality = self.data_info['laterality'].iloc[index]

        img, img_path = self.__getimg(path)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(image=np.array(img))["image"]

        pid = str(pid)
        exam_id = str(exam_id)

        return {'pid': pid, 'exam_id': exam_id, 'view': view, 'laterality': laterality, 'img_path': img_path,
                'img': img, 'age': age, 'density': density, 'race': race,
                'malignancy': malignancy, 'birads': birads,
                'years_to_cancer': years_to_cancer, 'years_to_last_followup': years_to_last_followup}

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