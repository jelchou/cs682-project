import torch
import pandas as pd 
import numpy as np
import torch.nn as nn
import torch.optim as optim

from meta_learning.dataset_loader import CUBDataset
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models


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
augmented_list = to_augment(train_dataset, ['sigmoid', 'blur', 'dropout'])
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
resnet50 = models.resnet50(pretrained=True).to(device)

# Freeze layers
for param in resnet50.parameters():
    param.requires_grad = False

num_of_classes = len(np.unique(train_dataset.labels['label']))

# Modify last layer
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, num_of_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50.fc.parameters(), lr=0.001, momentum=0.9)



# Training Loop
for epoch in range(10):
  resnet50.train()
  for inputs, labels in augmented_train_loader:
    inputs, labels = inputs.to(device).float(), labels.to(device).long()
    optimizer.zero_grad()
    outputs = resnet50(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

  # Training metrics
  train_loss, train_acc = metrics(resnet50, augmented_train_loader)
  train_losses.append(train_loss)
  train_accuracy.append(train_acc)

  # Validation metrics
  val_loss, val_acc = metrics(resnet50, baseline_val_loader)
  val_losses.append(val_loss)
  val_accuracy.append(val_acc)

  print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}')

# Save the model
torch.save(resnet50.state_dict(), './model_weights/resnet50_cub_baseline_7n.pth')

