import os
import scipy.io
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from PIL import Image

import numpy as np

# Import your transformation functions (e.g., apply_transform_crop)
from image_transforms import *


class OxfordFlowers(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.loader = default_loader
        self.transform = transform

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        label_mat = scipy.io.loadmat(os.path.join(root, 'imagelabels.mat'))
        self.labels = label_mat['labels'].squeeze() - 1
        split_mat = scipy.io.loadmat(os.path.join(root, 'setid.mat'))

        if split == 'train':
            self.indices = split_mat['trnid'].squeeze() - 1
        elif split == 'valid':
            self.indices = split_mat['valid'].squeeze() - 1
        elif split == 'test':
            self.indices = split_mat['tstid'].squeeze() - 1

        # Initialize an empty list for image transformations
        self.image_transformations = [None] * len(self.indices)
        
        # Initialize default transformation probabilities
        self.transform_probs = {'crop': 0.0, 'rotate': 0.0, 'rgb': 0.0,'blur':0.0,'sigmoid':0.0,'dropout':0.0}

    def set_transform_probs(self, transform_probs):
        self.transform_probs = transform_probs

    def apply_transform_policy(self, transform_policy):
        # num_images = len(self.filenames)
        num_images = len(self.indices)
        # Determine the number of images for each transformation
        num_crop = int(num_images * transform_policy['crop'])
        num_rotate = int(num_images * transform_policy['rotate'])
        num_rgb = int(num_images * transform_policy['rgb'])
        num_sigmoid = int(num_images * transform_policy['sigmoid'])
        num_blur = int(num_images * transform_policy['blur'])
        num_dropout = int(num_images * transform_policy['dropout'])

        # Create a list of transformations for the entire dataset
        transformations = ['crop'] * num_crop + ['rotate'] * num_rotate + ['rgb'] * num_rgb + ['sigmoid'] * num_sigmoid + ['blur'] * num_blur + ['dropout'] * num_dropout
        transformations += ['none'] * (num_images - len(transformations))  # Rest are 'none'

        # Shuffle the transformations
        np.random.shuffle(transformations)

        # Assign to each image
        self.image_transformations = transformations

    def __getitem__(self, idx):

        image_idx = self.indices[idx]
        image_path = os.path.join(self.root, 'jpg', f'image_{image_idx + 1:05d}.jpg')
        image = self.loader(image_path)
        label = self.labels[image_idx]


        #convert pil to numpy array
        image = np.array(image)
        # Apply transformations based on the policy
        transform_type = self.image_transformations[idx]
        if transform_type == 'crop':
            image = apply_transform_crop(image)
        elif transform_type == 'rotate':
            image = apply_transform_rotation(image)
        elif transform_type == 'rgb':
            image = apply_transform_rgb(image)
        elif transform_type == 'sigmoid':
            image = apply_transform_sigmoid(image)
        elif transform_type == 'blur':
             image = apply_transform_blur(image)
        elif transform_type == 'dropout':
             image = apply_transform_dropout(image)

        # image = cv2.resize(image, (224, 224))
        # # Normalize and adjust dimensions
        # image = normalize(image)
        # image = np.rollaxis(image, 2)  # Adjust shape to (C, H, W)
        # Convert back to PIL Image for torchvision transformations
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.indices)
