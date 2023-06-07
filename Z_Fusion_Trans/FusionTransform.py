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
csv_file = 'features.csv'
data = pd.read_csv(csv_file)


# Audio features from  1 to 61, and text data from  62
audio_features = data.iloc[0:,0:574]
print(audio_features)
text_data = data['text'].values


#Text embedding using BERT
model_name = 'bert-base-uncased'  # BERT model name
tokenizer = SentenceTransformer(model_name)
text_embeddings = tokenizer.encode(text_data, convert_to_tensor=True)


#Normalize audio features
audio_scaler = StandardScaler()
audio_features = audio_scaler.fit_transform(audio_features)


#Perform PCA on audio features
pca_audio = PCA(n_components=60)  # Adjust the number of components as needed
audio_embeddings = pca_audio.fit_transform(audio_features)


#Convert text embeddings to numpy array
text_embeddings = text_embeddings.cpu().numpy()
print(text_embeddings.shape)
print(audio_embeddings.shape)

fusion_embeddings = np.concatenate((audio_embeddings, text_embeddings), axis=1)
input_vector = fusion_embeddings


# Definining model & reducing dimension vector size = 128 rather than 828
model = tf.keras.Sequential([
    tf.keras.layers.Dense(600, activation='relu', input_shape=(828,)),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu')
])


for i in input_vector:    
    # Pass the input vector through the model
    print(i.shape)
    feature=(model.predict(np.expand_dims(i, axis=0)))
    with open('fusion.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(feature)


