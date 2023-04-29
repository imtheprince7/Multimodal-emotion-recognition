import librosa

# Load audio file
def Sound(path):
    audio_file = path #'path/to/audio.wav'
    y, sr = librosa.load(audio_file)

    # Extract MFCC features
    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc
