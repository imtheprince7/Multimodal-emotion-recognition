import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Dense
from keras.layers import Input, Reshape, Permute
from keras.layers import Attention, Dropout
import librosa
import cv2
from keras.applications.vgg16 import VGG16
# Step 1: Load audio and video files from separate folders
audio_folder = 'splitData/audioSplit/'
video_folder = 'splitData/videoSplit/'

audio_files = os.listdir(audio_folder)
video_files = os.listdir(video_folder)

# Step 2: Extract audio features (MFCC)
def extract_audio_features(audio_file):
    print(audio_file)
    audio, sr = librosa.load(audio_file, sr=None)  # Load audio file
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract MFCC features
    feature=[]
    for m in mfccs:
        feature.append(np.mean(m))
    return (feature)

    

audio_features = []
for audio_file in audio_files:
    features = extract_audio_features(os.path.join(audio_folder, audio_file))
    audio_features.append(features)

# Step 3: Extract video features (CNN)
def extract_video_features(video_file):
    print(video_file)
    from keras.models import Model
    from keras.applications.vgg16 import preprocess_input

    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

    cap = cv2.VideoCapture(video_file)  # Open video file
    success, frame = cap.read()  # Read the first frame

    if success:
        frame = cv2.resize(frame, (224, 224))  # Resize the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_array = np.expand_dims(frame, axis=0)
        img_array = preprocess_input(img_array)

        features = model.predict(img_array)
        return features.flatten()
    print(success)
    return None

video_features = []
for video_file in video_files:
    features = extract_video_features(os.path.join(video_folder, video_file))
    features=features.tolist()
    features.append(video_file.split('_')[0])
    features=np.array(features)
    video_features.append(features)

print(video_features)
print('*********************************************')
print(audio_features)

# Step 4: Combine audio and video features into a sequential representation
combined_features = []
for audio, video in zip(audio_features, video_features):
    combined_features.append(np.concatenate([audio, video]))
X = np.array(combined_features)
df=pd.DataFrame(X)
df.to_csv('AG.csv')




