import soundfile as sf
import numpy as np
from python_speech_features import mfcc

# Load the audio file
def Sound():
    audio, sr = sf.read("E:\Multimodal-emotion-recognition\Ses01F_impro01.wav")


# Convert to a mono signal
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)


    # Preprocess the audio signal
    audio = (audio - np.mean(audio)) / np.std(audio)
    mfcc_feat = mfcc(audio, sr)
    return mfcc_feat
# Print the MFCC feature vectors
print(Sound())

