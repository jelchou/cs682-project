import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Import transformations from image_transforms module
from image_transforms import read_image, apply_transform_crop, apply_transform_rotation, apply_transform_rgb, normalize

class CUBDataset(Dataset):
    def __init__(self, files_path, labels, train_test, image_name, train=True, transform_type=None):
        """
        Args:
            files_path (Path): Path to the dataset files.
            labels (pd.DataFrame): Dataframe containing labels.
            train_test (pd.DataFrame): Dataframe indicating train/test split.
            image_name (pd.DataFrame): Dataframe containing image file names.
            train (bool): Flag to indicate if the dataset is for training.
            transform_type (str): Type of transformation ('crop', 'rotate', 'rgb', None).
        """
        self.files_path = files_path
        self.labels = labels
        self.train_test = train_test
        self.image_name = image_name
        self.transform_type = transform_type

        if train:
          mask = self.train_test.is_train.values == 1

        else:
          mask = self.train_test.is_train.values == 0
          
        self.filenames = self.image_name.iloc[mask]
        self.labels = self.labels[mask]
        self.num_files = self.labels.shape[0]

    def set_transform_probs(self, transform_probs):
        self.transform_probs = transform_probs

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        y = self.labels.iloc[index, 1] - 1
        file_name = self.filenames.iloc[index, 1]
        path = self.files_path/'images'/file_name
        x = read_image(path)

        # Apply transformations
        if self.transform_type == 'crop':
            x = apply_transform_crop(x)
        elif self.transform_type == 'rotate':
            x = apply_transform_rotation(x)
        elif self.transform_type == 'rgb':
            x = apply_transform_rgb(x)
        else:  # Default resize for no specific transformation
            x = cv2.resize(x, (224, 224))

        # Normalize and adjust dimensions
        x = normalize(x)
        x = np.rollaxis(x, 2)  # Adjust shape to (C, H, W)
        return x, y

def create_dataloader(files_path, labels, train_test, image_name, batch_size=64, train=True, transform_type=None, num_workers=2):
    """ 
    Creates and returns a DataLoader for the CUBDataset.

    Args:
        files_path (Path): Path to the dataset files.
        labels, train_test, image_name (pd.DataFrame): Dataframes for dataset setup.
        batch_size (int): Size of each data batch.
        train (bool): Flag to indicate if the DataLoader is for training.
        transform_type (str): Type of transformation to apply.
        num_workers (int): Number of subprocesses to use for data loading.
    """
    dataset = CUBDataset(files_path, labels, train_test, image_name, train, transform_type)
    print(len(dataset))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
