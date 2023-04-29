from Bert import bert
from SoundFile import Sound
import torch
import numpy as np

# Load text features and audio features
text_features = bert()
audio_features = Sound()


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
text_aggregated = sequence_aggregator(text_features)
audio_aggregated = sequence_aggregator(audio_features)

# Concatenate text and audio aggregated features
features = np.concatenate([text_aggregated, audio_aggregated])

# Convert features to PyTorch tensor
features = torch.tensor(features)
print(features.shape)
