import librosa
import cv2

# Load audio signal
def ExtractFeature(path,filename):
    print(path)
    y, sr = librosa.load(path)

# Computing_spectrogram
    spec = librosa.stft(y)

    # Compute the magnitude spectrogram
    mag_spec = librosa.magphase(spec)[0]

    # Convert to decibels
    db_spec = librosa.amplitude_to_db(mag_spec)

    # Trim the spectrogram to a fixed length
    fixed_length_spec = librosa.util.fix_length(data=db_spec,size=1025)

    # Reshape the spectrogram to a 2D matrix
    spec_2d = fixed_length_spec.T
    spec_2d%=255
    cv2.imwrite(filename,spec_2d)