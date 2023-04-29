import numpy as np
from Bert import bert
from SoundFile import Sound
import numpy as np
import pandas as pd
import os


# Load text features and audio features
pathText='text/'
pathAudio='audio/'

# tempPathText=os.listdir(pathText)
# for i in tempPathText:
#     fullPath=os.path.join(pathText,i)
#     text_features = bert(fullPath)
text_features=[]
audio_features=[]
text_features.append(bert('E:\\Multimodal-emotion-recognition\\text\\Ses01F_impro01.txt'))

tempPathAudio=os.listdir(pathAudio)
for i in tempPathAudio:
    fullPath=os.path.join(pathAudio,i)
    audio_features.append(Sound(fullPath))


text_features=np.array(text_features)
audio_features=np.array(audio_features)
audio_features=audio_features.reshape(1,-1)
print(text_features.shape)
print(audio_features.shape)
text_feature_dim= text_features.shape
audio_feature_dim=audio_features.shape

# Define the sequence lengths for each modality
text_seq_len = len(text_features)
audio_seq_len =len(audio_features)

# Define the maximum sequence length for each modality
max_text_seq_len = max(text_features)
max_audio_seq_len = max(audio_features)

# Pad the sequences to their maximum length with zeros
text_features_padded = np.zeros((len(text_features), max_text_seq_len, text_feature_dim))
audio_features_padded = np.zeros((len(audio_features), max_audio_seq_len, audio_feature_dim))

for i in range(len(text_features)):
    text_features_padded[i, :text_seq_len[i], :] = text_features[i]

for i in range(len(audio_features)):
    audio_features_padded[i, :audio_seq_len[i], :] = audio_features[i]

# Concatenate the padded sequences along the time dimension
features_concatenated = np.concatenate([text_features_padded, audio_features_padded], axis=1)

# Perform sequence aggregation (e.g. mean pooling)
features_aggregated = np.mean(features_concatenated, axis=1)

df=pd.DataFrame(features_aggregated)
df.to_csv('Agg.csv')
