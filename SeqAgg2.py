from Bert import bert
from SoundFile import Sound
import torch
import numpy as np
import os

# Load text features and audio features





# Define sequence aggregator function
def sequence_aggregator(features):
    """
    Computes mean and max pooling over time for a given feature sequence.
    
    Args:
        features (numpy.ndarray): Feature sequence of shape (seq_len, feature_dim).
    
    Returns:
        numpy.ndarray: Aggregated features of shape (2 * feature_dim,).
    """
    mean_pooling = np.mean(features, axis=0)
    max_pooling = np.max(features, axis=0)
    return np.concatenate([mean_pooling, max_pooling])

# Apply sequence aggregator to text and audio features
pathText='text/'
pathAudio='audio/'
text_features=[]
tempPathText=os.listdir(pathText)
c=0
for i in tempPathText:
    c=c+1
    if(c>2):break
    fullPath=os.path.join(pathText,i)
    text_features.append(bert(fullPath))
text_aggregated = sequence_aggregator(text_features)
audio_features=[]
tempPathText=os.listdir(pathAudio)
c=0
for i in tempPathText:
    c=c+1
    if(c>2):break
    fullPath=os.path.join(pathAudio,i)
    audio_features.append(Sound(fullPath).reshape(-1))
print(text_features)
text_aggregated = sequence_aggregator(text_features)

# print(audio_features)
audio_aggregated = sequence_aggregator(audio_features)

# Concatenate text and audio aggregated features
features = np.concatenate([text_aggregated, audio_aggregated])

# Convert features to PyTorch tensor
features = torch.tensor(features)
print(features)
