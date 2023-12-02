import json
from dataset_loader import create_dataloader, CUBDataset
from model_setup import setup_resnet50
from model_training import *
from visualization import *
from utils import save_model, load_model

import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Subset, DataLoader


# Configuration and Paths
dataset_path = Path('./dataset/CUB_200_2011')
batch_size = 64
num_epochs = 5
num_classes = 200  # Set the correct number of classes based on your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset Metadata
labels = pd.read_csv(dataset_path/"image_class_labels.txt", header=None, sep=" ")
labels.columns = ["id", "label"]
train_test = pd.read_csv(dataset_path/"train_test_split.txt", header=None, sep=" ")
train_test.columns = ["id", "is_train"]
images = pd.read_csv(dataset_path/"images.txt", header=None, sep=" ")
images.columns = ["id", "name"]

model_name = "resnet50_fulltraining"

def main():

    
    # Load the augmentation policy
    with open('augmentation_policy.json', 'r') as f:
        augmentation_policy = json.load(f)

    # Create a full training DataLoader
    full_train_dataset = CUBDataset(dataset_path, labels, train_test, images, train=True)

    # Apply the policy to the dataset
    full_train_dataset.set_transform_probs(augmentation_policy)
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Create DataLoaders
    val_loader = create_dataloader(dataset_path, labels, train_test, images, batch_size, train=False, transform_type=None, num_workers=2)
    # Setup Model
    model = setup_resnet50(num_classes=num_classes, pretrained=True, freeze_layers=True)
    model.to(device)

    # Loss and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # Train the Model
    model, train_losses, val_losses, train_accuracy, val_accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

    # Save the Trained Model
    save_model(model, f'./model_weights/{model_name}.pth')

    # Load the Model (for evaluation)
    model = load_model(model, f'./model_weights/{model_name}.pth', device)

    # Evaluate the Model
    model.eval()

    top1 ,top5 = calc_accuracy(model, val_loader)
    print("top1 avg", top1.avg)
    print("top5 avg",top5.avg)

    # Visualization
    plot_training_results(train_losses, val_losses, train_accuracy, val_accuracy)

    # Generate classification report and confusion matrix (if needed)
    all_labels, all_preds = gather_all_predictions(val_loader, model, device)
    cm = confusion_matrix(all_labels, all_preds)
    classification_report_details(all_labels, all_preds)
    plot_confusion_matrix(all_labels, all_preds)

    #top confused pairs -10
    top_rows, top_cols, top_values = get_top_confused_classes(cm, num_pairs=10)
    top_confused_pairs = list(zip(top_rows, top_cols, top_values))


if __name__ == '__main__':
    main()