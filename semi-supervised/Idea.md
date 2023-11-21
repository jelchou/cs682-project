# Semi-Supervised Data Augmentation for Image Classification

## Introduction
This document outlines a semi-supervised approach to data augmentation for enhancing image classification tasks. The method integrates generative adversarial networks (GANs) and feature similarity measures to generate and validate augmented images, ensuring they are both diverse and representative of the original dataset.

## Workflow Overview
The workflow involves several stages, from preparing the dataset and generative models to integrating augmented images into the training process and refining the augmentation policy based on model performance.

## Step-by-Step Guide

### Step 1: Dataset Preparation
- **Acquire Dataset**: Obtain the Caltech-UCSD Birds (CUB) dataset.
- **Dataset Splitting**: Divide the dataset into training, validation, and testing sets, ensuring a balanced representation.
- **Additional Unlabeled Data**: If available, gather more unlabeled images to enhance the generative model training.

### Step 2: Designing Generative Models
- **Model Selection**: Choose between GANs or VAEs for generating images.
- **Transformation Focus**: Ensure the models generate realistic transformations relevant to bird species classification.
- **Training Generative Models**: Use unlabeled data or the non-labeled aspects of the CUB dataset for training.

### Step 3: Classifier Model Selection
- **Choose a Classifier**: Typically, a convolutional neural network (CNN) is suitable for image classification.
- **Architecture Selection**: Consider architectures like ResNet or VGG, potentially using pre-trained models for fine-tuning (Resnet50)

### Step 4: Data Augmentation and Training
- **Integrate Augmented Images**: Develop a strategy to intelligently mix original and generated images (Informed-Mixup Method). Start with equal proportion, this proportion will be tuned like hyperparameters. 
- **Training the Classifier**: Train the classifier using the combined dataset, monitoring performance on the validation set.

- **Adjust Policy based on feedback**: Use validation set performance to adjust the mix ratio and methods of augmentation. An augmentation policy can be created here. 

- **Iterative Refinement**- Continuously refine the augmentation strategy based on model performance. 

### Step 5: Hyperparameter Tuning and Validation
- **Parameter Tuning**: Adjust parameters such as the ratio of original to augmented images, learning rate, and batch size.
- **Validation Monitoring**: Use the validation set to fine-tune the model and avoid overfitting.

### Step 6: Evaluation and Testing
- **Model Evaluation**: Assess the classifier's performance on the test set using accuracy, precision, recall, and F1-score.
- **Baseline Comparison**: Compare results with a baseline model trained without augmented data.

### Step 7: Iteration and Improvement
- **Iterative Refinement**: Continuously refine the generative models and classifier based on test results.
- **Performance Analysis**: Analyze errors and adjust the augmentation strategy accordingly.

## Quality Assurance Steps
- **GAN Critic for Quality Check**: Use the GAN's discriminator to ensure generated images are close to the real data distribution.
- **Feature Similarity Check**: Implement cosine similarity measures using a pre-trained network to compare features of original and generated images.

## Conclusion
This semi-supervised approach aims to enhance the performance of image classification models, especially in scenarios with limited labeled data, by generating realistic and diverse training examples through GANs and ensuring their quality and relevance through feature similarity checks.


## look into
- feature similarity package
- https://pypi.org/project/image-similarity-measures/
- https://medium.com/scrapehero/exploring-image-similarity-approaches-in-python-b8ca0a3ed5a3
- https://www.geeksforgeeks.org/measure-similarity-between-images-using-python-opencv/
- https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
- https://www.kaggle.com/code/gauravduttakiit/generative-adversarial-networks-gans
- https://arxiv.org/abs/2202.08937#:~:text=Importantly%2C%20for%20most%20of%20the,of%20discriminative%20computer%20vision%20models

## work to-do

1. Create the Generative Model to generate dataset 
 -- look into pre-trained model / make a model from scratch
 -- input - CUB dataset / flower dataset
2. Data augmentation part - create the tuning of combination of dataset
 -- overfit a model with smaller training data, take the feedback and find the best policy. 
3. resnet part is ready

## Writeup - 
