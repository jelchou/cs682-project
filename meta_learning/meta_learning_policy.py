from dataset_loader import create_dataloader, CUBDataset
from model_setup import setup_resnet50
from model_training import *

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import random
import json

# Configuration and Paths
# dataset_path = Path('./dataset/CUB_200_2011')
dataset_path= Path('/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/dataset/CUB_200_2011')
batch_size = 64
num_epochs = 10
num_classes = 200  # Set the correct number of classes based on your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "resnet50_metalearning"
save_path = Path('/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/visualizations/meta_learning')


def main():
    # Load Dataset Metadata
    labels = pd.read_csv(dataset_path/"image_class_labels.txt", header=None, sep=" ")
    labels.columns = ["id", "label"]
    train_test = pd.read_csv(dataset_path/"train_test_split.txt", header=None, sep=" ")
    train_test.columns = ["id", "is_train"]
    images = pd.read_csv(dataset_path/"images.txt", header=None, sep=" ")
    images.columns = ["id", "name"]

    # Create a full training DataLoader

    # Create two instances of the dataset
    original_dataset = CUBDataset(dataset_path, labels, train_test, images, train=True)


    # full_train_dataset = CUBDataset(dataset_path, labels, train_test, images, train=True)
    val_loader = create_dataloader(dataset_path, labels, train_test, images, batch_size, train=False, transform_type=None, num_workers=2)

    '''
    Each parameter (p1, p2, p3, p4, p5, p6) is optimized independently within the range [0.01, 1]. 
    This means the optimization process is finding the best values for each parameter individually,
    without considering their sum.
    we are normalizign the probabilites after optimization in phase two code
    '''

    # Define the space of probabilities for Bayesian optimization
    space = [Real(0.01, 1, name='p1'), Real(0.01, 1, name='p2'), Real(0.01, 1, name='p3'),
             Real(0.01, 1, name='p4'),Real(0.01, 1, name='p5'),Real(0.01, 1, name='p6')]

    @use_named_args(space)
    def objective(**params):
        # Choose a subset of the dataset for this iteration
        subset_size = 2000  # Adjust the subset size
        # Create two subsets of the original dataset
        indices_subset1 = np.random.choice(len(original_dataset), subset_size, replace=False)
        indices_subset2 = np.random.choice(len(original_dataset), subset_size, replace=False)

        subset1 = Subset(original_dataset, indices_subset1)
        subset2 = Subset(original_dataset, indices_subset2)

        # Print current probability values
        print(f"Current probabilities: {params}")

        # Apply the current augmentation policy to the subset
        current_augmentation_policy = {'crop': params['p1'], 'rotate': params['p2'], 'rgb': params['p3'],'dropout':params['p4'],'blur':params['p5'],'sigmoid':params['p6']}
    
        # Train and evaluate the model
        model = setup_resnet50(num_classes=num_classes, pretrained=True, freeze_layers=True)
        model.to(device)
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        _, _, _, _, val_accuracy = train_model(model, subset1,subset2,val_loader, current_augmentation_policy, criterion,optimizer,save_path, num_epochs, device=device)
        # Print validation accuracy
        print(f"Validation accuracy for this iteration: {val_accuracy[-1]}")

        return -val_accuracy[-1]  # Maximize the last epoch's validation accuracy

    # Perform Bayesian optimization to find the best augmentation policy
    result = gp_minimize(objective, space, n_calls=50)
    best_probabilities = result.x
    

    # Normalize the probabilities so they sum up to 1
    total = sum(best_probabilities)
    best_probabilities = [value / total for value in best_probabilities]

    print("Best Probabilities:", best_probabilities)

    # Save the best augmentation policy to a JSON file
    augmentation_policy = {'crop': best_probabilities[0], 'rotate': best_probabilities[1], 'rgb': best_probabilities[2],'dropout':best_probabilities[3],'blur':best_probabilities[4],'sigmoid':best_probabilities[5]}
    with open('augmentation_policy_10epochperminimization_2000.json', 'w') as f:
        json.dump(augmentation_policy, f)

if __name__ == '__main__':
    main()