import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath('MultiModal/'))
import AudioFeaturesVideo
import VideoExtration
from transformers import BertTokenizer, BertModel
import torch
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('MultiModal\\fusion_embeddings.csv')
df=df['label']
LE=LabelEncoder()
LE.fit_transform(df)
def Feature(text,videopath,audiopath):
    audiofeature=AudioFeaturesVideo.extract_features(audiopath)
    print(audiofeature)
    print('AUDIO  Portion Done.................')
    videofeature=VideoExtration.feature(videopath)
    print(videofeature)
    print('VIDEO  Portion Done.................')
    fusion_data = np.concatenate((audiofeature, videofeature))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    text_embedding = torch.mean(output.last_hidden_state, dim=1).squeeze().numpy()
    fusion_embeddings = np.concatenate((fusion_data, text_embedding))
    print()
    print('TEXT Portion Done.................')
    model=load_model('MultiModal/FullyConnected.h5')
    fusion_embeddings=fusion_embeddings.reshape(1,-1)
    res= LE.inverse_transform([np.argmax(model.predict(fusion_embeddings))])
    print(res)
    return res



