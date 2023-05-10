import wave
import pandas as pd


def cut_audio(input_file, start_time, end_time, output_file):
    
    with wave.open(input_file, 'rb') as audio_file:

#audio file properties
        sample_width = audio_file.getsampwidth()
        num_channels = audio_file.getnchannels()
        frame_rate = audio_file.getframerate()
        num_frames = audio_file.getnframes()

#frame_rate_calculation      
        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)
        

        if end_frame > num_frames:
            end_frame = num_frames
        
# calculating_no_of_frame_to_read
        num_frames_to_read = end_frame - start_frame
        audio_file.setpos(start_frame)

        audio_data = audio_file.readframes(num_frames_to_read)
        
# Creating_new_wav_File_with_extracted_audio_segment
        with wave.open(output_file, 'wb') as output_audio_file:
            output_audio_file.setnchannels(num_channels)
            output_audio_file.setsampwidth(sample_width)
            output_audio_file.setframerate(frame_rate)
            output_audio_file.writeframes(audio_data)
 



df=pd.read_csv('dataSet/dataset_label/completeData.csv')
#print(df.head())

start=round(df['start_time'])
print(start.head())
end=round(df['end_time'])
print(end.head())
emotion=df['emotion']
print(emotion.head())
filesnames=df['file_name']  
print(df.head())

c=0
file=[]

for name in filesnames:
    try:
        c=c+1
        cut_audio('dataSet/audio/'+name+'.wav', start[c], end[c], 'splitData/audioSplit/output'+str(c)+'.wav')
        file.append('output'+str(c)+'.wav')
    except:
        file.append('')

df['audioSplitFilename']=file
df.to_csv('dataSet\dataset_label\completeData.csv')

