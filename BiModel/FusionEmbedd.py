import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# Load the CSV file
data = pd.read_csv('BiModel\\features1.csv')

# Extract audio, video, and text columns
audio_data = data.iloc[:, :61].values
text_data = data['text'].values

# Perform audio and video fusion (example: concatenate)
#fusion_data = np.concatenate((audio_data, video_data), axis=1)
print("Audion video Concanetaed")

# Perform text embedding using BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text_embeddings = []
for text in text_data:
    print(text)
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    text_embedding = torch.mean(output.last_hidden_state, dim=1).squeeze().numpy()
    text_embeddings.append(text_embedding)

text_embeddings = np.array(text_embeddings)

# Save fusion embeddings into a .npy file
fusion_embeddings = np.concatenate((audio_data, text_embeddings), axis=1)
np.save('BiModel/fusion_embeddings.npy', fusion_embeddings)
