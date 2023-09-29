#=======================================================================
# Section 1 : Libraries Import Section
#=======================================================================

print("Importing Libraries")
import torch
from dalle_pytorch import DiscreteVAE, DALLE
from torch.utils.data import Dataset, DataLoader
# from VAE_Encoder import CustomDataset
#Libraries for dataset section
import cv2
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#Libraries for text processing
import nltk
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from collections import Counter
# Downloading NLTK data if needed
nltk.download("punkt")
import pickle

#

#

def _preprocess_image(image_path):
    original_image = cv2.imread(image_path)
    # Resizing to the target image size
    resized_image = cv2.resize(original_image, (256, 256))
    # Normalizing pixel values to the range [0, 1]
    normalized_image = (resized_image / 255.0).astype(np.float32)
    return normalized_image

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


#================================================================
# Section 2 : Loading the saved values
#=======================================================================
save_dir = '/Users/varshasrinivas/Desktop/SJSU/Fall 2023/Dalle/text-to-image_generation'

print("Loading the saved train and test image dataloaders")
with open(os.path.join(save_dir, 'train_dataloader.pkl'), 'rb') as f:
    train_dataloader = pickle.load(f)

with open(os.path.join(save_dir, 'test_dataloader.pkl'), 'rb') as f:
    test_dataloader = pickle.load(f)

print("Loading the saved train and test image files")
with open(os.path.join(save_dir, 'train_files.pkl'), 'rb') as f:
    train_files = pickle.load(f)

with open(os.path.join(save_dir, 'test_files.pkl'), 'rb') as f:
    test_files = pickle.load(f)

# Load train and test datasets (optional, as they can be reconstructed from dataloaders and files)
# with open(os.path.join(save_dir, 'train_dataset.pkl'), 'rb') as f:
#     train_dataset = pickle.load(f)

# with open(os.path.join(save_dir, 'test_dataset.pkl'), 'rb') as f:
#     test_dataset = pickle.load(f)

print("Loading the pre-trained VAE encoder")
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

# Loading the saved VAE model's state dictionary
vae.load_state_dict(torch.load("vae_model.pth"))
print("Loaded pre trained model")

#=======================================================================
# Section 3 :Loading the captions file and pre-processing the text data
#=======================================================================


def preprocess_captions(captions_df):
    # Tokenize and preprocess train captions
    def tokenize_caption(caption):
        tokens = word_tokenize(caption)
        return tokens

    captions_df['tokens'] = captions_df['caption'].apply(tokenize_caption)

    # Counting token frequencies
    token_counts = Counter()
    for tokens in captions_df['tokens']:
        token_counts.update(tokens)

    # Printing vocabulary information
    vocab_size = len(token_counts)
    most_common_tokens = token_counts.most_common(50)
    print(f"Vocabulary size: {vocab_size}")
    print("Most common tokens:")
    for token, count in most_common_tokens:
        print(f"{token}: {count}")

    return captions_df

dataset_root = '/Users/varshasrinivas/Desktop/SJSU/Fall 2023/Dalle/coco_dataset'

print("Loading the captions file")
captions_df = pd.read_csv(os.path.join(dataset_root, 'captions.txt'), header=None, delimiter=',')
captions_df.columns = ['image_id', 'caption']

# Prepending the directory path to 'image_id' in captions_df
captions_df['image_id'] = dataset_root + '/images/' + captions_df['image_id']

print("filtering captions DataFrame for training and test captions")
train_captions_df = captions_df[captions_df['image_id'].isin(train_files)]
test_captions_df = captions_df[captions_df['image_id'].isin(test_files)]

# Define most_common_tokens here
common_captions_df = pd.concat([train_captions_df, test_captions_df], ignore_index=True)
common_captions_df = preprocess_captions(common_captions_df)
# Extract all tokens and flatten the list
all_tokens = [token for tokens in common_captions_df['tokens'] for token in tokens]
most_common_tokens = Counter(all_tokens).most_common(50)

print("Preprocessing captions - train data")
train_captions_df = preprocess_captions(train_captions_df)
test_captions_df = preprocess_captions(test_captions_df)

#=======================================================================
# Section 9 : Creating Data loader for text
#=======================================================================

print("Creating data loader for text data")
class TextDataset(Dataset):
    def __init__(self, text_data, max_length):
        self.text_data = text_data
        self.max_length = max_length
    def __len__(self):
        return len(self.text_data)
    def __getitem__(self, idx):
        tokens = self.text_data.iloc[idx]['tokens']
        # Truncating or pad the tokens to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += ['<pad>'] * (self.max_length - len(tokens))
        # Converting tokens to a tensor of long integers
        tokens = [token2id.get(token, token2id['<unk>']) for token in tokens]
        return torch.tensor(tokens, dtype=torch.long)
# Creating a mapping from tokens to unique IDs
token2id = {token: idx for idx, (token, _) in enumerate(most_common_tokens)}
token2id['<pad>'] = len(token2id)
token2id['<unk>'] = len(token2id)
batch_size = 16
max_sequence_length = 128

# Creating an instance of the text dataset for training and test sets
train_text_dataset = TextDataset(train_captions_df, max_sequence_length)
test_text_dataset = TextDataset(test_captions_df, max_sequence_length)

# Creating DataLoaders for batching and shuffling for training and test sets
print("Creating data loader for train text data")
train_text_dataloader = DataLoader(train_text_dataset, batch_size=batch_size, shuffle=True)
print("Creating data loader for test text data")
test_text_dataloader = DataLoader(test_text_dataset, batch_size=batch_size, shuffle=True)

#=======================================================================
# Section 10 : Initialize dalle Decoder
#=======================================================================

print("Initializing Dall e model")
dalle = DALLE(
    dim = 1024,
    vae = vae,                  
    num_text_tokens = 10000,    
    text_seq_len = 256,         
    depth = 12,                 
    heads = 16,                 
    dim_head = 64,              
    attn_dropout = 0.1,        
    ff_dropout = 0.1            
)

# Defining optimizer
optimizer_dalle = torch.optim.Adam(vae.parameters(), lr=1e-3)

#=======================================================================
# Section 11 : Training DALL-E (Fine tuning)
#=======================================================================

print("Training the Dall-e model (Fine Tuning)")
num_dalle_epochs = 10  
for epoch in range(num_dalle_epochs):
    print(f"Epoch {epoch + 1}/{num_dalle_epochs}")
    for batch_text, batch_images in zip(train_text_dataloader, train_dataloader):
        
        # Assuming batch_text is your text data with shape (batch_size, 128)
        desired_length = 256

        # Pad the text tokens to the desired length
        if batch_text.size(1) < desired_length:
            padding = torch.zeros(batch_text.size(0), desired_length - batch_text.size(1))
            batch_text = torch.cat((batch_text, padding), dim=1)
        elif batch_text.size(1) > desired_length:
            # Truncate if longer than desired length
            batch_text = batch_text[:, :desired_length]
        
        # converting batch_text from FloatTensor to LongTensor
        batch_text = batch_text.long()
        batch_images = batch_images.permute(0, 3, 1, 2)
        # Forward pass through DALL-E with text and images
        loss = dalle(batch_text, batch_images, return_loss=True)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer_dalle.step()
    print(f"Epoch {epoch + 1}/{num_dalle_epochs}, Loss: {loss.item()}")
print("Saving the Dalle decoder model")
torch.save(dalle.decoder.state_dict(), "dalle_decoder.pth")