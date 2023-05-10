import librosa
import numpy as np

def AudioFeature(path):
    y, sr = librosa.load(path, sr=None)

# Calculating_MFCC_features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    data=[]
    for j in mfcc:
        for k in j:
            if(k!=''):
                data.append(k)

# Shape_of_MFCC_feature_matrix
    return data

