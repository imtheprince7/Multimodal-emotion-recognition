import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import torch


# Read the audio features and text data from the CSV file
csv_file = 'z__FUSION/features.csv'
data = pd.read_csv(csv_file)


# Assuming audio features are stored in columns 1 to 68, and text data is in column 69
audio_features = data.iloc[0:,0:61]
print(audio_features)
text_data = data['text'].values


#Perform text embedding using BERT
model_name = 'bert-base-uncased'  # BERT model name
tokenizer = SentenceTransformer(model_name)
text_embeddings = tokenizer.encode(text_data, convert_to_tensor=True)

#Normalize audio features
audio_scaler = StandardScaler()
audio_features = audio_scaler.fit_transform(audio_features)


#Perform PCA on audio features
pca_audio = PCA(n_components=60)  # Adjust the number of components as needed
audio_embeddings = pca_audio.fit_transform(audio_features)

# Step 5: Convert text embeddings to numpy array
text_embeddings = text_embeddings.cpu().numpy()
print(text_embeddings.shape)
print(audio_embeddings.shape)

# Concatenate audio and text embeddings
fusion_embeddings = np.concatenate((audio_embeddings, text_embeddings), axis=1)
np.save('Fusion',fusion_embeddings)
print('Save Success')
