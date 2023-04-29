
# Define paths to audio and text folders
audio_folder = 'audio/'
text_folder = 'text/'
# shape (5, 9851, 781)


import torch
import torch.nn as nn
from Bert import bert
from SoundFile import Sound
from SeqAgg2 import features

# Define the number of filters, kernel size, and stride for the convolution

num_filters = 32
kernel_size = 3
stride = 1

# Load text features and audio features
text_features = bert()
audio_features = Sound()

# Concatenate the audio and text features along the time axis
concatenated_features = torch.cat([audio_features, text_features], dim=1)

# Define a 1D convolution layer
conv1d_layer = nn.Conv1d(in_channels=concatenated_features.shape[1], out_channels=num_filters, kernel_size=kernel_size, stride=stride)

# Apply the convolution layer to the concatenated features
convolved_features = conv1d_layer(concatenated_features)

# Perform sequence aggregation on the convolved features (e.g., mean pooling or max pooling)
aggregated_features = convolved_features.max(dim=2)[0]
