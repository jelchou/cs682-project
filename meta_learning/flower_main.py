from dataset_loader import create_dataloader
from model_setup import setup_resnet50
from model_training import *
from visualization import *
from utils import *

import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

import scipy.io
import json

from flower_dataset_loader import OxfordFlowers
# Configuration and Paths
dataset_path = Path('/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/flowers/dataset')
batch_size = 64
num_epochs = 10
num_classes = 102 # Set the correct number of classes 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset Metadata
setid_mat_path = dataset_path / 'setid.mat'
label_mat_path= dataset_path / 'imagelabels.mat'
# Load the .mat file
split_mat = scipy.io.loadmat(str(setid_mat_path))
label_mat = scipy.io.loadmat(str(label_mat_path))


model_name = "resnet50_flower"
# augment_policy = {
#     'crop': 0.0, 
#     'rotate': 0.0, 
#     'rgb': 0.0,
#     'dropout': 0.0,
#     'blur': 0.0,
#     'sigmoid': 0.0
# }

save_path = Path('/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/visualizations/flowers')

def main():

    with open('augmentation_policy_flower.json', 'r') as f:
        augment_policy = json.load(f)

    # Create DataLoaders

    # Create the dataset object
    train_dataset = OxfordFlowers(root=dataset_path, split='train', transform=None)
    transformed_dataset = OxfordFlowers(root=dataset_path, split='train', transform=None)
    val_dataset = OxfordFlowers(root=dataset_path, split='valid', transform=None)

    # Create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    # Setup Model
    model = setup_resnet50(num_classes=num_classes, pretrained=True, freeze_layers=True)
    model.to(device)

    # Loss and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # Train the Model
    model, train_losses, val_losses, train_accuracy, val_accuracy = flower_train_model(model, train_dataset,transformed_dataset,val_loader, augment_policy, criterion, optimizer, save_path,num_epochs,device=device)

    # model, train_losses, val_losses, train_accuracy, val_accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

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
    # top_pairs = get_top_confused_classes(cm, class_names, num_pairs=10)
    # for pair in top_pairs:
    #     print(pair)

    #top confused pairs -10
    # top_rows, top_cols, top_values = get_top_confused_classes(cm, num_pairs=10)
    # top_confused_pairs = list(zip(top_rows, top_cols, top_values))
    # print(top_confused_pairs)
    plot_tsne(model, val_loader, device,model_name,save_path)
    precision_scores = precision_score(all_labels,all_preds, average=None)
    average_precision = np.mean(precision_scores)
    # Plot histogram
    plt.bar(range(len(precision_scores)), precision_scores)
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title('Precision per Class')
    plt.savefig(f'{save_path}/Precision per class.png')
    plt.close()


    plt.hist(precision_scores)
    plt.xlabel('Precision')
    plt.ylabel('Number of Classes')
    plt.title('Precision of Validation data with Resnet50')
    plt.savefig(f'{save_path}/Precision of Validation data with Resnet50.png')

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