from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

input_directory = 'splittedData/splittedAudio/'
output_directory = 'splittedData/preProccesAudio/'


os.makedirs(output_directory, exist_ok=True)

def reduce_noise(audio_clip):
    noise_threshold = -50.0
    print('Starting noise reduction')

    # Split the audio clip into segments on silence
    segments = split_on_silence(audio_clip, min_silence_len=1000, silence_thresh=noise_threshold)

    # Concatenate the non-silent segments into a new audio clip
    reduced_clip = segments[0]
    for segment in segments[1:]:
        reduced_clip += segment

    return reduced_clip

def normalize_audio(audio_clip):
    target_amplitude = -3.0   # target peak amplitude -3 dBFS
    print('Starting normalization')

    # Normalize the audio clip to the target amplitude
    normalized_clip = audio_clip.normalize(headroom=target_amplitude)

    return normalized_clip

# Get a list of all files in the input directory
file_list = os.listdir(input_directory)

# Process each file in the input directory
for file_name in file_list:
    if file_name.endswith('.wav'):
        file_path = os.path.join(input_directory, file_name)
        audio_clip = AudioSegment.from_wav(file_path)
        print(file_path)

        reduced_clip = reduce_noise(audio_clip)
        normalized_clip = normalize_audio(reduced_clip)


        preprocessed_file_path = os.path.join(output_directory, file_name)
        normalized_clip.export(preprocessed_file_path, format='wav')
