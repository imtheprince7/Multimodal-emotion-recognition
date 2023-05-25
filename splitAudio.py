import wave

def cut_audio(input_file, start_time, end_time, output_file):
    # Open the input audio file
    try:
     with wave.open(input_file, 'rb') as audio_file:
        # Get the audio file properties
        sample_width = audio_file.getsampwidth()
        num_channels = audio_file.getnchannels()
        frame_rate = audio_file.getframerate()
        num_frames = audio_file.getnframes()

        # Calculate the start and end frame indices based on the given start and end times
        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)
        
        # Make sure the end frame is within the bounds of the audio file
        if end_frame > num_frames:
            end_frame = num_frames
        
        # Calculate the number of frames to read
        num_frames_to_read = end_frame - start_frame
        
        # Set the audio file pointer to the start frame
        audio_file.setpos(start_frame)
        
        # Read the audio data
        audio_data = audio_file.readframes(num_frames_to_read)
        
        # Create a new WAV file with the extracted audio segment
        with wave.open(output_file, 'wb') as output_audio_file:
            output_audio_file.setnchannels(num_channels)
            output_audio_file.setsampwidth(sample_width)
            output_audio_file.setframerate(frame_rate)
            output_audio_file.writeframes(audio_data)
    except:
        print('###############################################')
        print(output_file)
        print('###############################################')
        pass
import pandas as pd
df=pd.read_csv('Dataset\\dataset_1000.csv')


start=round(df['start_time'])
end=round(df['end_time'])
emotion=df['emotion']
filesnames=df['file_name']

# filesnames=set(filesnames)
c=0
file=[]
try:
    for name in filesnames:
        print(str(c)+' '+name)
        c=c+1
        cut_audio('data/audio/'+name+'.wav', start[c], end[c], 'splittedData/splittedAudio/'+emotion[c]+'_'+str(c)+'_.wav')
        file.append('output'+str(c)+'.wav')
except ValueError as e:
    print(e)

