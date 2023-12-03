import torch
import pandas as pd 
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle 

from meta_learning.dataset_loader import CUBDataset
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models
from itertools import combinations
from meta_learning.model_setup import setup_resnet50
from meta_learning.model_training import train_model
# potential transform types 
 #('crop', 'rotate', 'rgb', 'sigmoid', 'blur', 'dropout', None)

def to_augment(original_dataset, augment_list):
    final_augment = [original_dataset]
    for aug_type in augment_list:
        aug_data = CUBDataset(PATH, labels, train_test, images, train= True, transform_type=aug_type)
        final_augment.append(aug_data)
    
    return tuple(final_augment)

# Function for calculating metrics
def metrics(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            sum_loss += loss.item() * labels.size(0)
    return sum_loss / total, correct / total


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = Path('./dataset/CUB_200_2011')
labels = pd.read_csv(PATH/"image_class_labels.txt", header=None, sep=" ")
labels.columns = ["id", "label"]
train_test = pd.read_csv(PATH/"train_test_split.txt", header=None, sep=" ")
train_test.columns = ["id", "is_train"]
images = pd.read_csv(PATH/"images.txt", header=None, sep=" ")
images.columns = ["id", "name"]

train_dataset = CUBDataset(PATH, labels, train_test, images, train= True,)

number_augment_methods = [2,3]
augmentation_types = ['crop', 'rotate', 'rgb', 'sigmoid', 'blur', 'dropout']
grid_search = []

for num_augs in number_augment_methods:
    combination_list = list(combinations(augmentation_types, num_augs+1))
    for i in combination_list:
        grid_search.append(i)

results_validation = {}
best_val = -1 
best_resnet = None 



for augs in grid_search:
    print("Augmentation policy:")
    print(augs)
    augmented_list = to_augment(train_dataset, augs)
    concat_train_dataset = ConcatDataset(augmented_list)
    augmented_train_loader = DataLoader(concat_train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_dataset = CUBDataset(PATH, labels, train_test, images, train= False, transform= True)
    baseline_val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=2)

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []

    # Model Architecture
    curr_model = models.resnet50(pretrained=True).to(device)

    # Freeze layers
    for param in curr_model.parameters():
        param.requires_grad = False

    num_of_classes = len(np.unique(train_dataset.labels['label']))

    # Modify last layer
    num_ftrs = curr_model.fc.in_features
    curr_model.fc = nn.Linear(num_ftrs, num_of_classes).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(curr_model.fc.parameters(), lr=0.001, momentum=0.9)



        # Training Loop
    for epoch in range(10):
        curr_model.train()
        for inputs, labels in augmented_train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            outputs = curr_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Training metrics
        train_loss, train_acc = metrics(curr_model, augmented_train_loader)
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)

        # Validation metrics
        val_loss, val_acc = metrics(curr_model, baseline_val_loader)
        val_losses.append(val_loss)
        val_accuracy.append(val_acc)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}')
    # best validation at final epoch 
    final_validation_acc = val_acc

    if final_validation_acc > best_val:
        best_resnet = curr_model
    results_validation[augs] = final_validation_acc

with open('results_validation_grid_search.pkl', 'wb') as fp:
    pickle.dump(results_validation, fp)
    print('dictionary saved successfully to file')
torch.save(curr_model.state_dict(), './model_weights/resnet50_cub_grid_search.pth')

