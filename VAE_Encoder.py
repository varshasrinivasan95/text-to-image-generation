#=======================================================================
# Section 1 : Libraries Import Section
#=======================================================================

print("Importing Libraries")
import torch
from dalle_pytorch import DiscreteVAE, DALLE
#Libraries for dataset section
import cv2
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
#Libraries for text processing
import nltk
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from collections import Counter
# Downloading NLTK data if needed
nltk.download("punkt")
import pickle

#=======================================================================
# Section 2 : Image dataset loading 
#=======================================================================
if __name__ == "__main__":

    def _preprocess_image(image_path):
            original_image = cv2.imread(image_path)
            # Resizing to the target image size
            resized_image = cv2.resize(original_image, (256, 256))
            # Normalizing pixel values to the range [0, 1]
            normalized_image = (resized_image / 255.0).astype(np.float32)
            return normalized_image

    print("Loading image dataset")
    # Defining the path to the dataset
    dataset_root = '/Users/varshasrinivas/Desktop/SJSU/Fall 2023/Dalle/coco_dataset'
    # Creating a list to store image file paths
    image_files = []
    # Traverse the "images" directory and its subdirectories to find image files
    images_dir = os.path.join(dataset_root, 'images')
    for root, _, files in os.walk(images_dir):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, filename))

    #=======================================================================
    # Section 3 : Image pre-processing
    #=======================================================================

    print("Preprocessing image dataset")
    # Randomly selecting 5 image files
    random.seed(42)
    selected_image_files = random.sample(image_files, 5)
    # plotting the images
    fig, axes = plt.subplots(5, 3, figsize=(15, 15))
    axes[0, 0].set_title("Original Sample")
    axes[0, 1].set_title("Resized Sample")
    axes[0, 2].set_title("Normalized Sample")

    # Process and display the selected images
    for i, image_file in enumerate(selected_image_files):
        original_image = cv2.imread(image_file)

        # printing original image
        axes[i, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[i, 0].axis('off')
        
        # Resizing image to target size 
        target_size = (256, 256)
        resized_image = cv2.resize(original_image, target_size)
        axes[i, 1].imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        axes[i, 1].axis('off')
        
        # Normalizing pixel values to the range [0, 1]
        normalized_image = (resized_image * 255).astype(np.uint8)
        axes[i, 2].imshow(cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB))
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


    #=======================================================================
    # Section 4 : Train-test split and Creating Data loader
    #=======================================================================

    print("Splitting train and test data")
    # Split the dataset into training and testing sets
    train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

    print("Creating data loader")
    # Defining custom dataset class
    class CustomDataset(Dataset):
        def __init__(self, image_files):
            self.image_files = image_files
            
        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            # Loading and preprocess the image
            image_path = self.image_files[idx]
            image = _preprocess_image(image_path)
            return image

    # Define batch size and create a DataLoader
    batch_size_dataloader = 16 
    # Create an instance of custom dataset
    train_dataset = CustomDataset(train_files)
    test_dataset = CustomDataset(test_files)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_dataloader, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_dataloader, shuffle=False)

    print(f"Size of train_dataloader: {len(train_dataloader)} batches")
    print(f"Size of test_dataloader: {len(test_dataloader)} batches")

    #=======================================================================
    # Section 5 : Initialize VAE Encoder
    #=======================================================================

    print("Initializing the Discrete VAE (encoder)")

    vae = DiscreteVAE(
        channels=3,
        image_size = 256,
        num_layers = 3,           
        num_tokens = 8192,        
        codebook_dim = 512,      
        hidden_dim = 64,         
        num_resnet_blocks = 1,    
        temperature = 0.9,        
        straight_through = False, 
    )

    # Defining optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    #=======================================================================
    # Section 6 : Training VAE
    #=======================================================================

    print("Training VAE model")
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in train_dataloader:
            optimizer.zero_grad()
            # Forward pass through VAE
            #print("old shape",batch.shape)
            batch = torch.reshape(batch, [batch.size(0), 3, 256, 256])
            # batch = torch.reshape(batch, [32, 3, 256, 256])
            #print("new shape",batch.shape)
            loss = vae(batch, return_loss=True)
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    #=======================================================================
    # Section 7 : Saving the VAE model and other required variables
    #=======================================================================

    print("Saving the VAE encoder model")

    #Returns a dictionary with model's learnable parameters, saved to the "vae_model.pth" file
    torch.save(vae.state_dict(), "vae_model.pth")

    # Step 2: Training DALL-E with Pretrained VAE Encoder
    vae_encoder = vae.encoder  # Load the pretrained VAE encoder

    print("Saving other required variables for training Dall-e model")
    save_dir = '/Users/varshasrinivas/Desktop/SJSU/Fall 2023/Dalle/text-to-image_generation'

    # Saving train and test dataloaders
    with open(os.path.join(save_dir, 'train_dataloader.pkl'), 'wb') as f:
        pickle.dump(train_dataloader, f)

    with open(os.path.join(save_dir, 'test_dataloader.pkl'), 'wb') as f:
        pickle.dump(test_dataloader, f)

    # Save train and test files
    with open(os.path.join(save_dir, 'train_files.pkl'), 'wb') as f:
        pickle.dump(train_files, f)

    with open(os.path.join(save_dir, 'test_files.pkl'), 'wb') as f:
        pickle.dump(test_files, f)

    # Save train and test datasets (optional, as they can be reconstructed from dataloaders and files)
    with open(os.path.join(save_dir, 'train_dataset.pkl'), 'wb') as f:
        pickle.dump(train_dataset, f)

    with open(os.path.join(save_dir, 'test_dataset.pkl'), 'wb') as f:
        pickle.dump(test_dataset, f)

    print("End of VAE encoder")