import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import numpy as np
import torch


def plot_training_results(train_losses, val_losses, train_accuracy, val_accuracy,model_name,save_path):
    # save_path = Path('/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/visualizations')
    # Plotting
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Training and Validation Loss: Six Transformations")
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Training and Validation Accuracy:Six Transformations")
    plt.plot(train_accuracy, label="Training Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_path}/{model_name}_train_val_performance.png')
    plt.close()
    # plt.show()


def plot_confusion_matrix(labels, predictions,model_name,save_path):
    # save_path = Path('/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/visualizations')
    # Add code to plot confusion matrix
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(100, 100))  # You may want to adjust this based on how many classes you have
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'{save_path}/{model_name}_augmented_six_transform_confusion_matrix.png')
    plt.close()



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



def plot_tsne(model, dataloader,device,model_name,save_path):
    # save_path = Path('/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/visualizations')
    embeddings, labels = extract_embeddings(model, dataloader,device)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    # Add code to plot t-SNE visualization
    plt.figure(figsize=(10,10))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=5)
    legend = plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    # plt.add_artist(legend)
    plt.title("t-SNE visualization of feature embeddings")
    plt.savefig(f'{save_path}/{model_name}_six_transform_tsne.png')
    plt.close()

def classification_report_details(labels, predictions):
    # Generate classification report
    report = classification_report(labels, predictions)
    return report



# def get_top_confused_classes(confusion_matrix, num_pairs=10):
#     # Get indices of non-diagonal values (i.e., exclude true positives)
#     rows, cols = np.where(np.triu(confusion_matrix, 1) > 0)

#     # Get the values at these indices
#     confusions = confusion_matrix[rows, cols]

#     # Sort them in descending order
#     sorted_indices = np.argsort(confusions)[::-1]

#     # Get the top confused pairs and their values
#     top_rows = rows[sorted_indices][:num_pairs]
#     top_cols = cols[sorted_indices][:num_pairs]
#     top_values = confusions[sorted_indices][:num_pairs]

#     return top_rows, top_cols, top_values

def get_top_confused_classes(confusion_matrix, class_names_df, num_pairs=10):
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

    top_confused_pairs = []
    for i in range(num_pairs):
        row_class_name = class_names_df.loc[class_names_df['id'] == (top_rows[i] + 1), 'class_name'].values[0]
        col_class_name = class_names_df.loc[class_names_df['id'] == (top_cols[i] + 1), 'class_name'].values[0]
        value = top_values[i]
        top_confused_pairs.append((row_class_name, col_class_name, value))

    return top_confused_pairs


def visualize_transformations(dataset, epoch, transformations,save_path):
    # save_path = Path('/work/pi_hongyu_umass_edu/zonghai/mahbuba_medvidqa/semi_supervised/meta_learning/visualizations')
    fig, axs = plt.subplots(1, len(transformations), figsize=(15, 3))
    fig.suptitle(f'Epoch {epoch}: Sample Transformations')

    # Check if dataset is a Subset and access the original dataset's attributes
    if isinstance(dataset, torch.utils.data.Subset) and hasattr(dataset.dataset, 'image_transformations'):
        image_transformations = dataset.dataset.image_transformations
        original_dataset = dataset.dataset
    else:
        image_transformations = dataset.image_transformations
        original_dataset = dataset

    for i, trans_type in enumerate(transformations):
        # Find the index of the first occurrence of each transformation
        index = image_transformations.index(trans_type)
        image, _ = original_dataset[index]
        # Check if image is a PyTorch tensor, if so, convert to numpy
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        elif isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] == 3:
            # If it's a numpy array in (C, H, W) format, transpose it
            image = np.transpose(image, (1, 2, 0))
        
        #normalize
        image = (image - image.min()) / (image.max() - image.min())

        axs[i].imshow(image)
        axs[i].set_title(trans_type)
        axs[i].axis('off')

    plt.savefig(f'{save_path}/epoch_{epoch}_transformations.png')
    plt.close()


# def precision_score(true_labels, predictions,save_path):
#     # Calculate precision per class and average precision
#     precision_scores = precision_score(true_labels, predictions, average=None)
#     average_precision = np.mean(precision_scores)
#     # Plot histogram
#     plt.bar(range(len(precision_scores)), precision_scores)
#     plt.xlabel('Class')
#     plt.ylabel('Precision')
#     plt.title('Precision per Class')
#     plt.savefig(f'{save_path}/Precision per class.png')
#     plt.close()


#     plt.hist(precision_scores)
#     plt.xlabel('Precision')
#     plt.ylabel('Number of Classes')
#     plt.title('Precision of Validation data with Resnet50')
#     plt.savefig(f'{save_path}/Precision of Validation data with Resnet50.png')

#     return average_precision


def visualize_transformations_flowers(dataset, epoch, transformations, save_path):
    fig, axs = plt.subplots(1, len(transformations), figsize=(15, 3))
    fig.suptitle(f'Epoch {epoch}: Sample Transformations')

    # Check if dataset is a Subset and access the original dataset's attributes
    if isinstance(dataset, torch.utils.data.Subset) and hasattr(dataset.dataset, 'image_transformations'):
        image_transformations = dataset.dataset.image_transformations
        dataset_to_use = dataset.dataset
    else:
        image_transformations = dataset.image_transformations
        dataset_to_use = dataset

    for i, trans_type in enumerate(transformations):
        # Find the index of the first occurrence of each transformation
        index = image_transformations.index(trans_type)
        image, _ = dataset_to_use[index]

        # Convert image to correct format for imshow
        if isinstance(image, torch.Tensor):
            # If image is a PyTorch tensor, convert to numpy and transpose
            image = image.numpy().transpose((1, 2, 0))
        elif isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] == 3:
            # If image is a numpy array in (C, H, W) format, transpose it
            image = image.transpose((1, 2, 0))

        # Normalize image for display
        image = (image - image.min()) / (image.max() - image.min())

        axs[i].imshow(image)
        axs[i].set_title(trans_type)
        axs[i].axis('off')

    plt.savefig(f'{save_path}/epoch_{epoch}_transformations.png')
    plt.close()




# Usage Example:
# visualize_transformations(transformed_dataset, 1, ['crop', 'rotate', 'rgb'], '/path/to/save')
