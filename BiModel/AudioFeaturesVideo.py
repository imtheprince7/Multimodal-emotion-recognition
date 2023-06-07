import os
import csv
import librosa
import numpy as np
import pandas as pd

def extract_features(audio_path):
    print(audio_path)
    y, sr = librosa.load(audio_path)

    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y))
    rms_energy = np.mean(librosa.feature.rms(y=y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
    chroma_features = np.mean(librosa.feature.chroma_stft(y=y), axis=1)
    statistical_features = [np.mean(y), np.std(y), np.min(y), np.max(y), np.median(y)]
    tonnetz_features = np.mean(librosa.feature.tonnetz(y=y), axis=1)
    chroma_energy_normalized_features = np.mean(librosa.feature.chroma_cens(y=y), axis=1)
  

#Concatenating all features into a single array
    features = np.concatenate((
        [zero_crossing_rate, spectral_centroid, spectral_rolloff, spectral_contrast, spectral_bandwidth, rms_energy],
        mfcc, chroma_features, statistical_features, tonnetz_features,
        chroma_energy_normalized_features
    ))

    return features

# df1=pd.read_csv('Z_Fusion_Trans/attention_matrix.csv')
# audio_folder = 'splittedData\\splittedAudio\\'
# df=pd.read_csv('Dataset\\dataset_1000.csv')
# filenames=df['filenames']
# text=df['text']
# emotion=df['emotion']
# all_features = []
# label=[]
# f=[]
# # Iterate through audio files in the folder
# for i in range(len(df)):
#     if filenames[i].endswith('.wav'):  # Adjust the extensions as needed
#         try:
#             file_path = os.path.join(audio_folder, filenames[i])
#             features = extract_features(file_path).tolist()
#             features.append(text[i])
#             features.append(filenames[i])
#             label.append(emotion[i])
#             print(features)
            
#             all_features.append(features)
#         except:
#             print('#####################################')

# #Header in CSV file
# header = [
#     'Zero Crossing Rate', 'Spectral Centroid', 'Spectral Rolloff', 'Spectral Contrast', 'Spectral Bandwidth',
#     'Root Mean Square Energy'] + ['MFCC_{}'.format(i) for i in range(1, 21)] + ['Chroma_{}'.format(i) for i in range(1, 13)] + [
#              'Mean', 'Standard Deviation', 'Minimum', 'Maximum', 'Median'] + ['Tonnetz_{}'.format(i) for i in range(1, 7)] + [
#              'Chroma Energy Normalized_{}'.format(i) for i in range(1, 13)]+['text']+['filename']

# csv_file_path = 'features1.csv'

# # Write features to the CSV file
# with open(csv_file_path, 'w', newline='') as csv_file:
            
#     writer = csv.writer(csv_file)
    
#     writer.writerow(header)
    
# # Write the features for each audio file
#     for features in all_features:
#         writer.writerow(features)
# # df1['label']=label
# # df1.to_csv('attended_data.csv')

