from dataset_loader import create_dataloader
from model_setup import setup_resnet50
from model_training import *
from visualization import *
from utils import save_model, load_model

import torch
import pandas as pd
from pathlib import Path


# Configuration and Paths
dataset_path = Path('/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/dataset/CUB_200_2011')
batch_size = 64
num_epochs = 10
num_classes = 200  # Set the correct number of classes based on your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset Metadata
labels = pd.read_csv(dataset_path/"image_class_labels.txt", header=None, sep=" ")
labels.columns = ["id", "label"]
train_test = pd.read_csv(dataset_path/"train_test_split.txt", header=None, sep=" ")
train_test.columns = ["id", "is_train"]
images = pd.read_csv(dataset_path/"images.txt", header=None, sep=" ")
images.columns = ["id", "name"]

model_name = "resnet50_test"

def main():

    # Create DataLoaders
    train_loader = create_dataloader(dataset_path, labels, train_test, images, batch_size, train=True, transform_type=None, num_workers=2)
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
    save_model(model, f'/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/model_weights/{model_name}.pth')

    # Load the Model (for evaluation)
    model = load_model(model, f'/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/model_weights/{model_name}.pth', device)

    # Evaluate the Model
    model.eval()

    top1 ,top5 = calc_accuracy(model, val_loader, device)
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
    print(top_confused_pairs)


if __name__ == '__main__':
    main()