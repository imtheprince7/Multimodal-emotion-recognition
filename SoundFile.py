import soundfile as sf
import numpy as np
from python_speech_features import mfcc

# Load the audio file
audio, sr = sf.read("Ses01F_impro01.wav")


# Convert to a mono signal
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)
print("audio file",audio)

# Preprocess the audio signal
audio = (audio - np.mean(audio)) / np.std(audio)

# Extract the MFCC feature vectors
print("sr value:")
print(sr)
mfcc_feat = mfcc(audio, sr)

# Print the MFCC feature vectors
print("mfcc value:")
print(mfcc_feat)
print(mfcc_feat.shape)
