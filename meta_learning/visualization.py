import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import numpy as np
import torch

def plot_training_results(train_losses, val_losses, train_accuracy, val_accuracy):
    # Plotting
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Training and Validation Loss: Concat (4n) Dataset")
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Training and Validation Accuracy:Concat (4n) Dataset")
    plt.plot(train_accuracy, label="Training Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(labels, predictions, class_names):
    # Add code to plot confusion matrix
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(100, 100))  # You may want to adjust this based on how many classes you have
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('augmented_4n_confusion_matrix.png')
    plt.show()



def extract_embeddings(model, loader, device):
    embeddings = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in loader:
            # inputs = inputs.to(device)
            inputs = inputs.to(device).float()
            # Forward pass to get the embeddings before the final layer
            x = model.conv1(inputs)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            embedding = torch.flatten(x, 1)

            embeddings.append(embedding.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.vstack(embeddings), np.concatenate(labels_list)



def plot_tsne(features, labels,model_name, dataloader,device):

    embeddings, labels = extract_embeddings(model_name, dataloader,device)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    # Add code to plot t-SNE visualization
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=5)
    legend = plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    # plt.add_artist(legend)
    plt.title("t-SNE visualization of feature embeddings")
    plt.show()

def classification_report_details(labels, predictions):
    # Generate classification report
    report = classification_report(labels, predictions)
    return report



def get_top_confused_classes(confusion_matrix, num_pairs=10):
    # Get indices of non-diagonal values (i.e., exclude true positives)
    rows, cols = np.where(np.triu(confusion_matrix, 1) > 0)

    # Get the values at these indices
    confusions = confusion_matrix[rows, cols]

    # Sort them in descending order
    sorted_indices = np.argsort(confusions)[::-1]

    # Get the top confused pairs and their values
    top_rows = rows[sorted_indices][:num_pairs]
    top_cols = cols[sorted_indices][:num_pairs]
    top_values = confusions[sorted_indices][:num_pairs]

    return top_rows, top_cols, top_values

