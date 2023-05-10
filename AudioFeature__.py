import librosa
import numpy as np
import pandas as pd
import os
def zcr(data,frame_length=2048,hop_length=512):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)
   
  

def extract_features(data,sr,frame_length,hop_length):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length),
                      
                     ))
    return result
X=[]
audio_folder = 'Audio File/PD/'
for audio_file in os.listdir(audio_folder):
    audio_path = os.path.join(audio_folder, audio_file)
    data,sr=librosa.load(audio_path,duration=2.5,offset=0.6)
    
    aud=extract_features(data,sr,2048,512)
#   
    X.append(aud)
X=np.array(X)
print(X.shape)
X=pd.DataFrame(X)
X.to_csv("AudioFeatureData2.csv")
print('Done')
# data,sr=librosa.load('Audio File\HP\B1ACNUTCOC40M230320171121.wav',duration=2.5,offset=0.6)
    
# aud=extract_features(data,sr,2048,512)
# print(aud)