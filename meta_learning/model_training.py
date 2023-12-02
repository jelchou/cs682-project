import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import numpy as np

from utils import AverageMeter
from dataset_loader import CUBDataset

def train_model(model, train_set,augment_policy, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    train_losses, val_losses = [], []
    train_accuracy, val_accuracy = [], []

    for epoch in range(num_epochs):
        #dynamic transformation for each epoch
        # different images will get different transformation in different epochs
        # Apply the policy to the underlying CUBDataset of the subset
        if hasattr(train_set, 'dataset'):
            if isinstance(train_set.dataset, CUBDataset):
                train_set.dataset.apply_transform_policy(augment_policy)

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

def metrics(model, dataloader, criterion, device='cuda'):
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)

        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def calc_accuracy(model, dataloader, device):
    model.eval()
    model.cuda()

    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (inputs, labels) in enumerate(dataloader):

        inputs, labels = inputs.cuda().float(), labels.cuda().long()
        # obtain the outputs from the model
        outputs = model.forward(inputs)
        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

    return top1 ,top5


def gather_all_predictions(val_loader, model, device='cuda'):
    # Store all predictions and true labels
    all_preds = []
    all_labels = []

    # Loop through validation data
    with torch.no_grad(): # No need to compute gradients when evaluating
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()

            # Forward pass
            outputs = model(inputs)

            # Get the predicted class labels
            _, preds = torch.max(outputs, 1)

            # Append to lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return all_labels, all_preds
