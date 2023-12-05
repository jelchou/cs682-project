import json
from dataset_loader import create_dataloader, CUBDataset
from model_setup import setup_resnet50
from model_training import *
from visualization import *
from utils import *

import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Subset, DataLoader


# Configuration and Paths
dataset_path = Path('/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/dataset/CUB_200_2011')
batch_size = 32
num_epochs = 20
num_classes = 200  # Set the correct number of classes based on your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset Metadata
labels = pd.read_csv(dataset_path/"image_class_labels.txt", header=None, sep=" ")
labels.columns = ["id", "label"]
train_test = pd.read_csv(dataset_path/"train_test_split.txt", header=None, sep=" ")
train_test.columns = ["id", "is_train"]
images = pd.read_csv(dataset_path/"images.txt", header=None, sep=" ")
images.columns = ["id", "name"]

class_names = pd.read_csv(dataset_path/"classes.txt", header=None, sep=" ")
class_names.columns = ["id", "class_name"]

model_name = "resnet50_fulltraining_2n_bsz32_epochs20"
save_path = Path('/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/visualizations/fulltrain2n_bsz32_epochs20')

def main():

    
    # Load the augmentation policy
    with open('augmentation_policy_10epochperminimization_2000.json', 'r') as f:
        augment_policy = json.load(f)
    # Create a full training DataLoader
    full_train_dataset = CUBDataset(dataset_path, labels, train_test, images, train=True)
    transformed_dataset = CUBDataset(dataset_path, labels, train_test, images, train=True)

    print("Training with augmentation policy of 2000 subset")

    # Apply the policy to the dataset
    # full_train_dataset.apply_transform_policy(augmentation_policy)
    # full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Create DataLoaders
    val_loader = create_dataloader(dataset_path, labels, train_test, images, batch_size, train=False, transform_type=None, num_workers=2)
    # Setup Model
    model = setup_resnet50(num_classes=num_classes, pretrained=True, freeze_layers=True)
    model.to(device)

    # Loss and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
        # Print validation accuracy
    # Train the Model
    # model, train_losses, val_losses, train_accuracy, val_accuracy = train_model(model, full_train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)
    model, train_losses, val_losses, train_accuracy, val_accuracy = train_model(model, full_train_dataset,transformed_dataset,val_loader, augment_policy, criterion, optimizer, save_path,num_epochs,device=device)

    # Save the Trained Model
    save_model(model, f'/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/model_weights/{model_name}.pth')

    # Load the Model (for evaluation)
    model = load_model(model, f'/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/model_weights/{model_name}.pth', device)

    # Evaluate the Model
    model.eval()

    top1 ,top5 = calc_accuracy(model, val_loader,device)
    print("top1 avg", top1.avg)
    print("top5 avg",top5.avg)

    # Visualization
    plot_training_results(train_losses, val_losses, train_accuracy, val_accuracy,model_name,save_path)

    # Generate classification report and confusion matrix (if needed)
    all_labels, all_preds = gather_all_predictions(val_loader, model, device)
    cm = confusion_matrix(all_labels, all_preds)
    print(classification_report_details(all_labels, all_preds))
    
    plot_confusion_matrix(all_labels, all_preds,model_name,save_path)

    #top confused pairs
    top_pairs = get_top_confused_classes(cm, class_names, num_pairs=10)
    for pair in top_pairs:
        print(pair)

    #top confused pairs -10
    # top_rows, top_cols, top_values = get_top_confused_classes(cm, num_pairs=10)
    # top_confused_pairs = list(zip(top_rows, top_cols, top_values))
    # print(top_confused_pairs)
    plot_tsne(model, val_loader, device,model_name,save_path)

    average_precision= precision_score(all_labels, all_preds,save_path)
    print(f"Average Precision: {average_precision}")


    # Per-class accuracy calculation
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    average_accuracy = np.mean(class_accuracies)

    # Print per-class accuracy
    for idx, acc in enumerate(class_accuracies):
        print(f"Accuracy for class {idx}: {acc}")

    print(f"Average Accuracy: {average_accuracy}")


if __name__ == '__main__':
    main()