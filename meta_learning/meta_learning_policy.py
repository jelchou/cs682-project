from dataset_loader import create_dataloader, CUBDataset
from model_setup import setup_resnet50
from model_training import *
from visualization import plot_training_results, plot_confusion_matrix, classification_report_details, get_top_confused_classes
from utils import save_model, load_model

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
dataset_path = Path('./dataset/CUB_200_2011')
batch_size = 64
num_epochs = 5
num_classes = 200  # Set the correct number of classes based on your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "resnet50_test"

def main():
    # Load Dataset Metadata
    labels = pd.read_csv(dataset_path/"image_class_labels.txt", header=None, sep=" ")
    labels.columns = ["id", "label"]
    train_test = pd.read_csv(dataset_path/"train_test_split.txt", header=None, sep=" ")
    train_test.columns = ["id", "is_train"]
    images = pd.read_csv(dataset_path/"images.txt", header=None, sep=" ")
    images.columns = ["id", "name"]

    # Create a full training DataLoader
    full_train_dataset = CUBDataset(dataset_path, labels, train_test, images, train=True)
    # full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = create_dataloader(dataset_path, labels, train_test, images, batch_size, train=False, transform_type=None, num_workers=2)

    # Phase 1: Policy Generation
    # Define the space of probabilities
    space = [Real(0, 1, name='p1'), Real(0, 1, name='p2'), Real(0, 1, name='p3')]

    @use_named_args(space)
    def objective(**params):
        subset_size = 1000  # Adjust the subset size
        indices = np.random.choice(len(full_train_dataset), subset_size, replace=False)
        subset = Subset(full_train_dataset, indices)
        subset.dataset.set_transform_probs({'crop': params['p1'], 'rotate': params['p2'], 'rgb': params['p3']})
        subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)

        # Train and evaluate the model
        model = setup_resnet50(num_classes=num_classes, pretrained=True, freeze_layers=True)
        model.to(device)
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        _, _, _, _, val_accuracy = train_model(model, subset_loader, val_loader, criterion, optimizer, num_epochs=1, device=device)
        return -val_accuracy[-1]  # Maximize the last epoch's validation accuracy

    # Perform Bayesian optimization
    result = gp_minimize(objective, space, n_calls=50)
    best_probabilities = result.x
    print("Best Probabilities:", best_probabilities)

    # Save the policy to a JSON file
    augmentation_policy = {'crop': best_probabilities[0], 'rotate': best_probabilities[1], 'rgb': best_probabilities[2]}
    with open('augmentation_policy.json', 'w') as f:
        json.dump(augmentation_policy, f)

if __name__ == '__main__':
    main()
