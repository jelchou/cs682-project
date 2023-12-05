import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import numpy as np
from utils import *
from dataset_loader import CUBDataset
from visualization import visualize_transformations, visualize_transformations_flowers
from flower_dataset_loader import OxfordFlowers
from torch.utils.data import ConcatDataset, DataLoader


def train_model(model, original_dataset,transformed_dataset, val_loader, augment_policy, criterion, optimizer, save_path,num_epochs=10, batch_size=64, num_workers=2, device='cuda'):
    train_losses, val_losses = [], []
    train_accuracy, val_accuracy = [], []

    print(f"original dataset{len(original_dataset)}, transformed_dataset{len(transformed_dataset)}")

    for epoch in range(num_epochs):
        # Apply the policy to the underlying CUBDataset of the subset
        if hasattr(transformed_dataset, 'dataset') and isinstance(transformed_dataset.dataset, CUBDataset):
            transformed_dataset.dataset.apply_transform_policy(augment_policy)
            # Visualize or save transformations
            visualize_transformations(transformed_dataset, epoch, ['crop', 'rotate', 'rgb', 'sigmoid', 'blur', 'dropout'],save_path)

        
        # Concatenate the original and transformed datasets
        concatenated_dataset = ConcatDataset([original_dataset, transformed_dataset])

        print(f"concatenated dataset len{len(concatenated_dataset)}")

        # Recreate the DataLoader to reflect the updated dataset
        train_loader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # dataset.apply_transform_policy(augment_policy)
        model.train()
        print('Training loop..')
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


        # Training metrics
        train_loss, train_acc = metrics(model, train_loader,criterion, device)

        # Validation metrics
        val_loss, val_acc = metrics(model,val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
        val_losses.append(val_loss)
        val_accuracy.append(val_acc)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}')
    
    return model, train_losses, val_losses, train_accuracy, val_accuracy






def flower_train_model(model, original_dataset, transformed_dataset, val_loader, augment_policy, criterion, optimizer, save_path, num_epochs=10, batch_size=64, num_workers=2, device='cuda'):
    train_losses, val_losses = [], []
    train_accuracy, val_accuracy = [], []

    print(f"Original dataset size: {len(original_dataset)}, Transformed dataset size: {len(transformed_dataset)}")

    for epoch in range(num_epochs):
        # Apply the policy to the underlying OxfordFlowers dataset of the subset
        if hasattr(transformed_dataset, 'dataset') and isinstance(transformed_dataset.dataset, OxfordFlowers):
            transformed_dataset.dataset.apply_transform_policy(augment_policy)
            # Visualize or save transformations (if needed)
            # visualize_transformations_flowers(transformed_dataset, epoch, ['crop', 'rotate', 'rgb', 'sigmoid', 'blur', 'dropout'], save_path)

        # Concatenate the original and transformed datasets
        concatenated_dataset = ConcatDataset([original_dataset, transformed_dataset])

        print(f"Concatenated dataset size: {len(concatenated_dataset)}")

        # Recreate the DataLoader to reflect the updated dataset
        train_loader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        model.train()
        print('Training loop..')
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Training metrics
        train_loss, train_acc = metrics(model, train_loader, criterion, device)

        # Validation metrics
        val_loss, val_acc = metrics(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
        val_losses.append(val_loss)
        val_accuracy.append(val_acc)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}')
    
    return model, train_losses, val_losses, train_accuracy, val_accuracy


