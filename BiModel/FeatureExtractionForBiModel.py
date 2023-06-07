import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath('BiModel/'))
import AudioFeaturesVideo
from transformers import BertTokenizer, BertModel
import torch
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('BiModel\\fusion_embeddings.csv')
df=df['label']
LE=LabelEncoder()
LE.fit_transform(df)
def Feature(text,audiopath):
    audiofeature=AudioFeaturesVideo.extract_features(audiopath)
    print(audiofeature)
    print('AUDIO  Portion Done.................')


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    text_embedding = torch.mean(output.last_hidden_state, dim=1).squeeze().numpy()
    fusion_embeddings = np.concatenate((audiofeature, text_embedding))
    print()
    print('TEXT Portion Done.................')

#Model Loading which i have saved earlier
    model=load_model('BiModel/FullyConnected.h5')  # Accuracy was 55%.
    fusion_embeddings=fusion_embeddings.reshape(1,-1)
    result= LE.inverse_transform([np.argmax(model.predict(fusion_embeddings))])
    print(result)
# returning result for Flask server
    return result



