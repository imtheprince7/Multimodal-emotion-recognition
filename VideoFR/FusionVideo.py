import csv
import torch
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# audio features and text data from the CSV file
csv_file = 'features1.csv'
data = pd.read_csv(csv_file)


# Assuming audio features are stored in columns 1 to 68, and text data is in column 69
audio_features = data.iloc[0:,0:573]
print(audio_features)
text_data = data['text'].values

#Perform text embedding using BERT
model_name = 'bert-base-uncased'  # BERT model name
tokenizer = SentenceTransformer(model_name)
text_embeddings = tokenizer.encode(text_data, convert_to_tensor=True)

#Normalize audio features
audio_scaler = StandardScaler()
audio_features = audio_scaler.fit_transform(audio_features)

#Perform PCA on audio_video features
pca_audio = PCA(n_components=573)  # Adjust the number of components as needed
audio_video_embeddings = pca_audio.fit_transform(audio_features)

#Convert text embeddings to numpy array
text_embeddings = text_embeddings.cpu().numpy()
print(text_embeddings.shape)
print(audio_video_embeddings.shape)

fusion_embeddings = np.concatenate((audio_video_embeddings, text_embeddings), axis=1)
input_vector = fusion_embeddings


# Defining model to reduce dimension vector size =128 rather than 828
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, activation='relu', input_shape=(1341,)),
    tf.keras.layers.Dense(800, activation='relu'),
    tf.keras.layers.Dense(600, activation='relu'),
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu')
])
data=[]
c=0

for i in input_vector:
    feature=(model.predict(np.expand_dims(i, axis=0)).tolist())
    with open('fusion.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(feature)




