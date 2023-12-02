import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

# Import transformations from image_transforms module
from image_transforms import read_image, apply_transform_crop, apply_transform_rotation, apply_transform_rgb, apply_transform_blur, apply_transform_dropout, apply_transform_sigmoid, normalize

class CUBDataset(Dataset):
    def __init__(self, files_path, labels, train_test, image_name, train=True, transform_type=None):
        """
        Args:
            files_path (Path): Path to the dataset files.
            labels (pd.DataFrame): Dataframe containing labels.
            train_test (pd.DataFrame): Dataframe indicating train/test split.
            image_name (pd.DataFrame): Dataframe containing image file names.
            train (bool): Flag to indicate if the dataset is for training.
            transform_type (str): Type of transformation ('crop', 'rotate', 'rgb', 'sigmoid', 'blur', 'dropout', None).
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
        # Initialize an empty list for image transformations
        self.image_transformations = [None] * len(self.filenames)

    def apply_transform_policy(self, transform_policy):
        num_images = len(self.filenames)
        # Determine the number of images for each transformation
        num_crop = int(num_images * transform_policy['crop'])
        num_rotate = int(num_images * transform_policy['rotate'])
        num_rgb = int(num_images * transform_policy['rgb'])

        # Create a list of transformations for the entire dataset
        transformations = ['crop'] * num_crop + ['rotate'] * num_rotate + ['rgb'] * num_rgb
        transformations += ['none'] * (num_images - len(transformations))  # Rest are 'none'

        # Shuffle the transformations
        np.random.shuffle(transformations)

        # Assign to each image
        self.image_transformations = transformations

    def set_transform_probs(self, transform_probs):
        self.transform_probs = transform_probs

    def __len__(self):
        return self.num_files

    #for predetermined  transformation for each image
    def __getitem__(self, index):
        y = self.labels.iloc[index, 1] - 1
        file_name = self.filenames.iloc[index, 1]
        path = self.files_path/'images'/file_name
        x = read_image(path)

        ## apply the predetermined transformations
        transform_type = self.image_transformations[index]
        # Apply the predetermined transformation
        transform_type = self.image_transformations[index]
        if transform_type == 'crop':
            x = apply_transform_crop(x)
        elif transform_type == 'rotate':
            x = apply_transform_rotation(x)
        elif transform_type == 'rgb':
            x = apply_transform_rgb(x)
        elif transform_type == 'sigmoid':
            x = apply_transform_sigmoid(x)
        elif transform_type == 'blur':
             x = apply_transform_blur(x)
        elif transform_type == 'dropout':
             x = apply_transform_dropout(x)
        else:  # Default resize for no specific transformation
            x = cv2.resize(x, (224, 224))

        # Normalize and adjust dimensions
        x = normalize(x)
        x = np.rollaxis(x, 2)  # Adjust shape to (C, H, W)
        return x, y


    '''
    applies a transformation at the individual image level based on the specified probabilities, 
    rather than applying a transformation to a fixed percentage of the entire dataset
    randomly selects a transformation for each individual image every time it is accessed, 
    based on the specified probabilities
    '''
    # def __getitem__(self, index):
    #     y = self.labels.iloc[index, 1] - 1
    #     file_name = self.filenames.iloc[index, 1]
    #     path = self.files_path/'images'/file_name
    #     x = read_image(path)

    #     # Decide transformation based on probabilities
    #     transform_choices = ['crop', 'rotate', 'rgb', 'none']  # Include 'none' for no transformation
    #     transform_probabilities = [self.transform_probs.get(t, 0) for t in transform_choices]
    #     chosen_transform = random.choices(transform_choices, weights=transform_probabilities, k=1)[0]

    #     if chosen_transform == 'crop':
    #         x = apply_transform_crop(x)
    #     elif chosen_transform == 'rotate':
    #         x = apply_transform_rotation(x)
    #     elif chosen_transform == 'rgb':
    #         x = apply_transform_rgb(x)
    #     # 'none' means no specific transformation

    #     # Normalize and adjust dimensions
    #     x = normalize(x)
    #     x = np.rollaxis(x, 2)  # Adjust shape to (C, H, W)
    #     return x, y
        

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
