import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Load the csv files with audio and text features
audio_df = pd.read_csv('audio_features.csv')
text_df = pd.read_csv('text_features.csv')

# Separate the features from the filenames
audio_features = np.array(audio_df.drop('rows', axis=1))
text_features = np.array(text_df.drop('rows', axis=1))

# Define the attention module
class AttentionModule(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, hidden_dim, output_dim):
        super().__init__()
        
        # Define the layers for the audio attention
        self.audio_att_fc1 = nn.Linear(input_dim_audio, hidden_dim)
        self.audio_att_fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Define the layers for the text attention
        self.text_att_fc1 = nn.Linear(input_dim_text, hidden_dim)
        self.text_att_fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Define the activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, audio_tensor, text_tensor):
        # Compute the attention weights for audio and text
        audio_att = self.sigmoid(self.audio_att_fc2(self.relu(self.audio_att_fc1(audio_tensor))))
        text_att = self.sigmoid(self.text_att_fc2(self.relu(self.text_att_fc1(text_tensor))))
        
        # Apply the attention weights to the features
        audio_weighted = audio_tensor * audio_att
        text_weighted = text_tensor * text_att
        
        # Concatenate the weighted features
        weighted_concat = torch.cat([audio_weighted, text_weighted], dim=1)
        
        # Compute the output of the attention module
        attention_output = torch.sum(weighted_concat, dim=1)
        
        return attention_output

# Initialize the attention module
input_dim_audio = audio_features.shape[1]
input_dim_text = text_features.shape[1]
hidden_dim = 64
output_dim = 1
attention_module = AttentionModule(input_dim_audio, input_dim_text, hidden_dim, output_dim)

# Convert the features to tensors
audio_tensor = torch.from_numpy(audio_features).float()
text_tensor = torch.from_numpy(text_features).float()

# Apply the attention module to the features
attention_output = attention_module(audio_tensor, text_tensor)

# Print the shape of the output
print(attention_output.shape)


# Convert the attention output tensor to a numpy array
attention_output_array = attention_output.detach().numpy()

# Create a pandas DataFrame with the attention output
attention_df = pd.DataFrame(attention_output_array, columns=['attention_output'])

# Write the DataFrame to a CSV file
attention_df.to_csv('attentionModel_Output.csv', index=False)
