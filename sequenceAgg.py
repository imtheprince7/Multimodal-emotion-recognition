import os
import pandas as pd
import librosa
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Define paths to audio and text folders
audio_folder = 'audio/'
text_folder = 'text/'

# Define number of MFCC coefficients and text feature vector size
n_mfcc = 13
text_max_len = 128

# Extract audio features
audio_features = []
for audio_file in os.listdir(audio_folder):
    audio_path = os.path.join(audio_folder, audio_file)
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    audio_features.append(mfcc.T)  # Transpose to have shape (time_steps, n_mfcc)

# Extract text features
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
text_features = []
for text_file in os.listdir(text_folder):
    text_path = os.path.join(text_folder, text_file)
    with open(text_path, 'r') as f:
        text = f.read().strip()
    encoded_text = tokenizer.encode_plus(text, max_length=text_max_len, padding='max_length', truncation=True, return_tensors='pt')
    outputs = model(**encoded_text)
    last_hidden_state = outputs.last_hidden_state
    text_feature_vector = last_hidden_state.mean(dim=1).detach().numpy()
    text_features.append(text_feature_vector)

# Sequence aggregation of features
combined_features = []
for i in range(min(len(audio_features), len(text_features))):
    audio_feats = audio_features[i]
    text_feat = text_features[i]
    combined_feats = np.concatenate([audio_feats, np.tile(text_feat, (audio_feats.shape[0], 1))], axis=1)
    combined_features.append(combined_feats)

# Pad features to the same length
max_length = max([feat.shape[0] for feat in combined_features])
padded_features = []
for feat in combined_features:
    padding_length = max_length - feat.shape[0]
    padded_feat = np.pad(feat, ((0, padding_length), (0, 0)), mode='constant')
    padded_features.append(padded_feat)

# Stack features into a single tensor
stacked_features = np.stack(padded_features)

# Display stacked features
data = stacked_features.reshape(5, 9851*781)
data = pd.DataFrame(data)
data.to_csv('csv_file')
